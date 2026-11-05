from __future__ import annotations
import json
import operator
import os
import re
import time
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

# ============================================================
# ENVIRONMENT AND LLM

# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        f"GROQ_API_KEY is missing. Add it to: {BASE_DIR / '.env'}"
    )
# Do not set temperature here: some reasoning models enforce
# provider-specific sampling settings.

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    max_tokens=1800,
)
# Groq free/on-demand TPM limits can be reached by parallel LangGraph workers.
# Serialize API calls and leave a small gap between requests.
_llm_call_lock = threading.Lock()
_last_llm_call = 0.0
_MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("GROQ_CALL_INTERVAL", "35"))

def _invoke_llm(messages, max_attempts: int = 5, model=None):
    global _last_llm_call
    for attempt in range(max_attempts):
        with _llm_call_lock:
            elapsed = time.monotonic() - _last_llm_call
            wait_for = _MIN_SECONDS_BETWEEN_CALLS - elapsed
            if wait_for > 0:
                time.sleep(wait_for)
            try:
                result = (model or llm).invoke(messages)
                _last_llm_call = time.monotonic()
                return result
            except Exception as exc:
                _last_llm_call = time.monotonic()
                message = str(exc).lower()
                is_request_too_large = (
                    "413" in message
                    or "request too large" in message
                    or "tokens per minute" in message
                )
                if is_request_too_large:
                    raise RuntimeError(
                        "Groq rejected this prompt because it exceeds the current "
                        "TPM/request budget. Reduce research evidence or prompt "
                        "length, then try again."
                    ) from exc
                is_rate_limit = (
                    "429" in message
                    or "rate_limit_exceeded" in message
                    or "rate limit" in message
                )
                if not is_rate_limit or attempt == max_attempts - 1:
                    raise
                retry_wait = min(30 * (attempt + 1), 90)
                time.sleep(retry_wait)
    raise RuntimeError("LLM request failed after retry attempts.")

# ============================================================
# PYDANTIC SCHEMAS

# ============================================================

class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: list[str] = Field(min_length=3, max_length=6)
    target_words: int = Field(default=180, ge=120, le=550)
    tags: list[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class BlogPlan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup",
        "comparison", "system_design"
    ] = "explainer"
    constraints: list[str] = Field(default_factory=list)
    tasks: list[Task] = Field(min_length=3, max_length=9)

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: list[str] = Field(default_factory=list)

class EvidenceItem(BaseModel):
    title: str
    url: str
    snippet: str = ""
    published_at: Optional[str] = None
    source: Optional[str] = None

class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str

class ImagePlan(BaseModel):
    markdown: str
    images: list[ImageSpec] = Field(default_factory=list)

class BlogState(TypedDict, total=False):
    topic: str
    as_of: str
    mode: str
    needs_research: bool
    queries: list[str]
    evidence: list[EvidenceItem]
    research_status: str
    enable_research: bool
    plan: Optional[BlogPlan]
    sections: Annotated[list[tuple[int, str]], operator.add]
    merged_md: str
    markdown_with_placeholders: str
    image_specs: list[dict]
    generated_images: list[dict]
    image_status: str
    image_count: int
    final: str
    generate_images: bool

# ============================================================
# JSON HELPERS

# ============================================================

def _message_text(content) -> str:
    """Convert an AIMessage content value into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)

def _extract_json_object(text: str) -> dict:
    """Extract a JSON object from plain text or Markdown-fenced model output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\\s*```\\s*$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    try:
        from json_repair import loads as repair_json
        parsed = repair_json(text)
        if isinstance(parsed, dict):
            return parsed
    except ImportError:
        pass
    except Exception:
        pass
    raise ValueError("The model did not return a valid JSON object.")

def _invoke_json_model(model_cls, system_prompt: str, human_prompt: str):
    """
    Ask Groq for JSON and validate it with Pydantic.
    A second attempt includes the exact schema and validation feedback.
    """
    schema_text = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    first_prompt = (
        system_prompt
        + "\n\nReturn one JSON object that conforms exactly to this JSON schema:\n"
        + schema_text
        + "\nReturn JSON only. Do not use Markdown fences."
    )
    last_error = None
    for attempt in range(2):
        prompt = first_prompt
        if attempt == 1:
            prompt += (
                "\n\nYour previous response did not validate. "
                "Correct the JSON keys and value types. "
                "Do not rename schema fields."
                f"\nPrevious validation error: {last_error}"
            )
        response = _invoke_llm([
            SystemMessage(content=prompt),
            HumanMessage(content=human_prompt),
        ])
        raw_text = _message_text(response.content)
        try:
            data = _extract_json_object(raw_text)
            return model_cls.model_validate(data)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
    raise ValueError(
        f"Groq could not produce valid {model_cls.__name__} JSON after 2 attempts. "
        f"Last error: {last_error}"
    )

def _normalize_blog_plan(data: dict, topic: str, mode: str) -> BlogPlan:
    """
    Handle common Groq naming variations while preserving the app's
    canonical BlogPlan/Task schema.
    """
    if "blog_title" not in data and "title" in data:
        data["blog_title"] = data.pop("title")
    if "tasks" not in data and "sections" in data:
        data["tasks"] = data.pop("sections")
    data.setdefault("blog_title", topic.title())
    data.setdefault("audience", "Readers interested in the requested topic")
    data.setdefault("tone", "Professional, clear, and engaging")
    data.setdefault("constraints", [])
    if "blog_kind" not in data:
        data["blog_kind"] = "news_roundup" if mode == "open_book" else "explainer"
    normalized_tasks = []
    for index, raw_task in enumerate(data.get("tasks", []), start=1):
        task = dict(raw_task)
        if "title" not in task and "heading" in task:
            task["title"] = task.pop("heading")
        task.setdefault("id", index)
        task.setdefault("title", f"Section {index}")
        task.setdefault("goal", f"Explain {task['title']} clearly and practically.")
        task.setdefault("target_words", 180)
        task.setdefault("tags", [])
        task.setdefault("requires_research", False)
        task.setdefault("requires_citations", False)
        task.setdefault("requires_code", False)
        # The Task schema requires 3-6 bullets. Pad short model outputs
        # with useful, non-empty prompts rather than failing validation.
        bullets = list(task.get("bullets") or [])
        while len(bullets) < 3:
            bullets.append(
                f"Explain an important concept, practical implication, or example "
                f"related to {task['title']}."
            )
        task["bullets"] = bullets[:6]
        normalized_tasks.append(task)
    data["tasks"] = normalized_tasks
    return BlogPlan.model_validate(data)

# ============================================================
# LLM ROUTER

# ============================================================

ROUTER_PROMPT = """
You are an AI blog research router.
Choose:
- closed_book for stable, evergreen topics.
- hybrid for topics that benefit from some current examples.
- open_book for latest news, pricing, regulations, or policies.
If research is needed, provide 3-6 focused search queries.
For evergreen topics, research may be unnecessary.
Never invent current events.
Return fields exactly as defined in the supplied RouterDecision schema:
needs_research, mode, reason, queries.
"""

def router_node(state: BlogState) -> dict:
    topic = state["topic"]
    as_of = state.get("as_of", date.today().isoformat())
    try:
        decision = _invoke_json_model(
            RouterDecision,
            ROUTER_PROMPT,
            f"Topic: {topic}\nDate: {as_of}",
        )
    except Exception:
        # Keep the workflow usable if routing JSON fails.
        decision = RouterDecision(
            needs_research=False,
            mode="closed_book",
            reason="Fallback routing because the router response was invalid.",
            queries=[],
        )
    # The sidebar research toggle takes priority. When enabled, always try
    # to gather sources, including for evergreen topics.
    enable_research = state.get("enable_research", True)
    queries = decision.queries[:6]
    if enable_research and not queries:
        queries = [
            f"{topic} overview and fundamentals",
            f"{topic} practical examples and latest developments",
            f"{topic} best practices and authoritative sources",
        ]
    return {
        "mode": decision.mode if enable_research else "closed_book",
        "needs_research": bool(enable_research),
        "queries": queries if enable_research else [],
    }

def route_after_router(state: BlogState) -> str:
    return "research" if state.get("needs_research") else "planner"

# ============================================================
# RESEARCH NODE (TAVILY WITH DUCKDUCKGO FALLBACK)

# ============================================================

def research_node(state: BlogState) -> dict:
    """
    Search with Tavily when configured; otherwise use the ddgs package.
    Tavily failures also fall back to DuckDuckGo.
    """
    queries = state.get("queries", [])[:6]
    if not queries:
        topic = state.get("topic", "")
        queries = [
            f"{topic} overview and fundamentals",
            f"{topic} practical examples and best practices",
        ]
    results = []
    provider_used = None
    provider_errors = []
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            search = TavilySearchResults(max_results=5)
            for query in queries:
                try:
                    results.extend(search.invoke({"query": query}) or [])
                except Exception as exc:
                    provider_errors.append(f"Tavily query failed: {exc}")
            if results:
                provider_used = "Tavily"
        except Exception as exc:
            provider_errors.append(f"Tavily unavailable: {exc}")
    # No Tavily key, no Tavily results, or Tavily failure: use DDGS.
    if not results:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                for query in queries:
                    try:
                        rows = ddgs.text(
                            query,
                            max_results=5,
                            safesearch="moderate",
                        )
                        results.extend(rows or [])
                    except Exception as exc:
                        provider_errors.append(f"DuckDuckGo query failed: {exc}")
            if results:
                provider_used = "DuckDuckGo (DDGS)"
        except ImportError:
            provider_errors.append(
                "The ddgs package is not installed. Run: "
                "python -m pip install -U ddgs"
            )
        except Exception as exc:
            provider_errors.append(f"DuckDuckGo search unavailable: {exc}")
    evidence: list[EvidenceItem] = []
    seen = set()
    for item in results:
        url = item.get("url") or item.get("href")
        if not url or url in seen:
            continue
        seen.add(url)
        evidence.append(
            EvidenceItem(
                title=item.get("title") or url,
                url=url,
                snippet=(
                    item.get("content")
                    or item.get("body")
                    or item.get("snippet")
                    or ""
                ),
                published_at=(
                    item.get("published_date")
                    or item.get("date")
                    or None
                ),
                source=item.get("source") or provider_used,
            )
        )
    if evidence:
        status = f"Found {len(evidence)} source(s) using {provider_used}."
    else:
        detail = " | ".join(provider_errors[-3:])
        status = "No research sources were found."
        if detail:
            status += " Details: " + detail
        else:
            status += (
                " Search providers returned no results. Check your internet "
                "connection or try a more specific topic."
            )
    # For open-book topics, keep recent dated sources. Undated results are
    # retained because search engines frequently omit publication dates.
    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state.get("as_of", date.today().isoformat()))
        cutoff = as_of - timedelta(days=7)
        filtered = []
        for item in evidence:
            if not item.published_at:
                filtered.append(item)
                continue
            try:
                published = date.fromisoformat(item.published_at[:10])
                if published >= cutoff:
                    filtered.append(item)
            except (ValueError, TypeError):
                filtered.append(item)
        evidence = filtered
    return {
        "evidence": evidence[:20],
        "research_status": status,
    }

# ============================================================
# BLOG PLANNER

# ============================================================

PLANNER_PROMPT = """
You are a professional blog content strategist.
Create a useful outline for the requested blog topic.
Return exactly one JSON object with these keys:
blog_title, audience, tone, blog_kind, constraints, tasks.
Create exactly 3 tasks to keep the response concise.
Each task must contain:
id, title, goal, bullets, target_words, tags,
requires_research, requires_citations, requires_code.
Rules:
- Each task must have exactly 3 short bullets.
- target_words should normally be 150.
- blog_kind must be one of: explainer, tutorial, news_roundup, comparison, system_design.
- Use concise strings. Do not include long explanations in the JSON.
- Do not invent current facts, sources, statistics, or citations.
- Return valid JSON only. No Markdown fences or commentary.
"""

def planner_node(state: BlogState) -> dict:
    topic = str(state["topic"]).strip()
    mode = state.get("mode", "closed_book")
    as_of = state.get("as_of", date.today().isoformat())
    # Only pass a few short source summaries to the planner.
    evidence = []
    for raw_item in state.get("evidence", [])[:3]:
        item = raw_item.model_dump() if isinstance(raw_item, EvidenceItem) else raw_item
        evidence.append({
            "title": str(item.get("title", ""))[:100],
            "url": str(item.get("url", ""))[:180],
            "snippet": str(item.get("snippet", ""))[:120],
        })
    human_prompt = (
        f"Topic: {topic[:300]}\n"
        f"Mode: {mode}\n"
        f"Date: {as_of}\n"
        f"Evidence: {json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}\n"
        "Create a concise 3-section outline. Keep every field short."
    )
    schema_text = json.dumps(BlogPlan.model_json_schema(), ensure_ascii=False)
    system_prompt = (
        PLANNER_PROMPT
        + "\n\nRequired JSON schema:\n"
        + schema_text
        + "\nReturn one complete JSON object only."
    )
    last_error = None
    plan = None
    json_llm = llm.bind(
        response_format={"type": "json_object"},
        max_tokens=1800,
        reasoning_effort="low",
        include_reasoning=False,
    )
    for attempt in range(3):
        prompt = system_prompt
        if attempt == 1:
            prompt += (
                "\nPrevious response was invalid. Return a shorter JSON object "
                "with exactly 3 tasks and exactly 3 short bullets per task. "
                "Do not include optional detail. "
                f"Validation/parser issue: {str(last_error)[:300]}"
            )
        elif attempt == 2:
            prompt = (
                "Return valid JSON only. No Markdown. "
                "Use keys blog_title, audience, tone, blog_kind, constraints, tasks. "
                "Make exactly 3 tasks. Each task keys: id,title,goal,bullets,"
                "target_words,tags,requires_research,requires_citations,requires_code. "
                "Each bullets array must contain exactly 3 short strings. "
                "blog_kind must be explainer, tutorial, news_roundup, comparison, "
                "or system_design."
            )
        response = _invoke_llm(
            [SystemMessage(content=prompt), HumanMessage(content=human_prompt)],
            model=json_llm,
        )
        raw_text = _message_text(response.content)
        try:
            # Use JSON mode response for planner; try standard parser then repair.
            raw_data = _extract_json_object(raw_text)
            if not isinstance(raw_data, dict):
                raise ValueError("Planner response was not a JSON object.")
            plan = _normalize_blog_plan(raw_data, topic, mode)
            break
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            # Attempt JSON repair before making another model call.
            try:
                from json_repair import loads as repair_json
                repaired = repair_json(raw_text)
                if isinstance(repaired, dict):
                    plan = _normalize_blog_plan(repaired, topic, mode)
                    break
            except ImportError:
                pass
            except Exception as repair_exc:
                last_error = repair_exc
    if plan is None:
        raise ValueError(
            "Groq could not create a valid blog plan after 3 attempts. "
            f"Last parser/validation error: {last_error}. "
            "Install JSON repair with: python -m pip install json-repair"
        )
    if mode == "open_book":
        plan.blog_kind = "news_roundup"
    return {"plan": plan}

WORKER_PROMPT = """
You are a senior technical writer.
Write ONE complete blog section in Markdown.
Rules:
- Start with a level-two Markdown heading (##).
- Cover every supplied bullet.
- Follow the specified audience and tone.
- Be practical, technically accurate, and detailed.
- Include code only when requested.
- Do not fabricate sources, quotations, benchmarks, or statistics.
- Cite supplied URLs for external factual claims.
- Do not write the entire blog; write only this section.
"""

def writer_node(payload: dict) -> dict:
    task = Task.model_validate(payload["task"])
    plan = BlogPlan.model_validate(payload["plan"])
    evidence = [
        EvidenceItem.model_validate(item)
        for item in payload.get("evidence", [])
    ]
    evidence_text = "\n".join(
        f"- {item.title}: {item.url} — {item.snippet[:220]}"
        for item in evidence[:4]
    ) or "No external research evidence was supplied."
    response = _invoke_llm([
        SystemMessage(content=WORKER_PROMPT),
        HumanMessage(content=(
            f"Blog: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n"
            f"Topic: {payload['topic']}\n"
            f"Section: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Target words: {task.target_words}\n"
            f"Bullets:\n- " + "\n- ".join(task.bullets) +
            f"\nRequires code: {task.requires_code}\n"
            f"Requires citations: {task.requires_citations}\n"
            f"Evidence:\n{evidence_text}"
        )),
    ])
    return {
        "sections": [(task.id, _message_text(response.content).strip())]
    }

# ============================================================
# MERGE CONTENT

# ============================================================

def merge_node(state: BlogState) -> dict:
    plan = state.get("plan")
    if not plan:
        raise ValueError("Plan missing.")
    sections = sorted(state.get("sections", []), key=lambda item: item[0])
    body = "\n\n".join(section for _, section in sections)
    markdown = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": markdown}

# ============================================================
# REALISTIC AI IMAGE GENERATION (POLLINATIONS)

# ============================================================

def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "blog"

def image_plan_node(state: BlogState) -> dict:
    """
    Create image prompts locally from the blog plan.
    This avoids spending additional Groq tokens on a separate image-planning call.
    """
    merged_md = state.get("merged_md", "")
    if not state.get("generate_images", False):
        return {
            "markdown_with_placeholders": merged_md,
            "image_specs": [],
            "image_status": "Image generation was not selected.",
        }
    pollinations_key = os.getenv("POLLINATIONS_API_KEY", "").strip()
    if not pollinations_key:
        return {
            "markdown_with_placeholders": merged_md,
            "image_specs": [],
            "image_status": (
                "POLLINATIONS_API_KEY is missing. Add a Pollinations secret key "
                "starting with sk_ to your .env file."
            ),
        }
    if not pollinations_key.startswith("sk_"):
        return {
            "markdown_with_placeholders": merged_md,
            "image_specs": [],
            "image_status": (
                "Invalid Pollinations key type. Use a secret key beginning "
                "with sk_. A pk_ App Key does not authenticate backend image "
                "generation requests."
            ),
        }
    plan = state.get("plan")
    if not plan or not plan.tasks:
        return {
            "markdown_with_placeholders": merged_md,
            "image_specs": [],
            "image_status": "No blog plan was available for image prompts.",
        }
    count = max(1, min(int(state.get("image_count", 2)), 3, len(plan.tasks)))
    slug = _slugify(plan.blog_title)
    markdown = merged_md
    specs = []
    for index, task in enumerate(plan.tasks[:count], start=1):
        placeholder = f"[[IMAGE_{index}]]"
        filename = f"{slug}-image-{index}"
        alt = f"Photorealistic editorial image illustrating {task.title}"
        caption = f"Visual context: {task.title}"
        prompt = (
            "Create a highly realistic, premium editorial photograph for a "
            "professional technology blog. "
            f"Main subject and concept: {plan.blog_title}. "
            f"Specific visual focus: {task.title}. "
            f"Context: {task.goal}. "
            "Show a believable real-world scene with physically plausible objects, "
            "natural materials, realistic human proportions if people are present, "
            "subtle depth of field, refined composition, cinematic but natural "
            "lighting, lifelike textures, professional DSLR photography, "
            "high detail, photorealistic, landscape 16:9 composition. "
            "For abstract software concepts, use a sophisticated real-world "
            "metaphorical scene rather than a fake screenshot or diagram. "
            "No text, no letters, no captions, no logos, no watermarks, "
            "no UI text, no collage."
        )
        specs.append({
            "placeholder": placeholder,
            "filename": filename,
            "alt": alt,
            "caption": caption,
            "prompt": prompt,
        })
        # Put the cover image immediately after the blog title.
        if index == 1:
            first_break = markdown.find("\n\n")
            if first_break >= 0:
                markdown = (
                    markdown[:first_break + 2]
                    + placeholder + "\n\n"
                    + markdown[first_break + 2:]
                )
            else:
                markdown += "\n\n" + placeholder + "\n"
        else:
            heading = f"## {task.title}"
            if heading in markdown:
                markdown = markdown.replace(
                    heading, heading + "\n\n" + placeholder, 1
                )
            else:
                markdown += "\n\n" + placeholder + "\n"
    return {
        "markdown_with_placeholders": markdown,
        "image_specs": specs,
        "image_status": f"Prepared {len(specs)} realistic image prompt(s).",
    }

def generate_pollinations_image(prompt: str) -> tuple[bytes, str]:
    """
    Generate one image through Pollinations' authenticated image endpoint.
    The secret API key stays on the backend and is never sent to Streamlit UI.
    """
    api_key = os.getenv("POLLINATIONS_API_KEY")
    if not api_key:
        raise RuntimeError("POLLINATIONS_API_KEY is missing from .env.")
    if not api_key.startswith("sk_"):
        raise RuntimeError(
            "Pollinations requires a backend secret key beginning with sk_. "
            "A pk_ App Key will return 401 Unauthorized."
        )
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The requests package is missing. Run: python -m pip install requests"
        ) from exc
    from urllib.parse import quote
    model = os.getenv(
        "POLLINATIONS_IMAGE_MODEL",
        "black-forest-labs/flux.1.1-pro",
    )
    encoded_prompt = quote(prompt, safe="")
    url = f"https://gen.pollinations.ai/image/{encoded_prompt}"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        params={
            "model": model,
            "width": 1280,
            "height": 704,
            "safe": "true",
        },
        timeout=(20, 240),
    )
    if response.status_code >= 400:
        try:
            provider_detail = response.text[:700]
        except Exception:
            provider_detail = "No error body returned."
        if response.status_code == 401:
            raise RuntimeError(
                "Pollinations returned 401 Unauthorized. Check that your "
                "POLLINATIONS_API_KEY is a valid sk_ secret key."
            )
        if response.status_code == 402:
            raise RuntimeError(
                "Pollinations returned 402 Payment Required. Your key is valid, "
                "but its Pollen balance/budget may be exhausted or insufficient."
            )
        if response.status_code == 403:
            raise RuntimeError(
                "Pollinations returned 403 Forbidden. Check that this key is "
                "allowed to use the selected image model."
            )
        raise RuntimeError(
            f"Pollinations HTTP {response.status_code}: {provider_detail}"
        )
    content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
    extension_by_type = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    extension = extension_by_type.get(content_type)
    if not extension:
        preview = response.text[:300] if response.text else "No response body."
        raise RuntimeError(
            f"Image provider returned unexpected content type '{content_type}'. "
            f"Response: {preview}"
        )
    if not response.content:
        raise RuntimeError("Image provider returned an empty image.")
    return response.content, extension

def place_images_node(state: BlogState) -> dict:
    markdown = (
        state.get("markdown_with_placeholders")
        or state.get("merged_md", "")
    )
    specs = state.get("image_specs", [])
    generated_images = []
    errors = []
    if specs and os.getenv("POLLINATIONS_API_KEY"):
        image_dir = BASE_DIR / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            placeholder = spec.get("placeholder", "")
            safe_stem = Path(spec.get("filename") or "blog-image").stem
            safe_stem = _slugify(safe_stem)
            try:
                image_bytes, extension = generate_pollinations_image(spec["prompt"])
                filename = safe_stem + extension
                output_path = image_dir / filename
                output_path.write_bytes(image_bytes)
                image_markdown = (
                    f"![{spec.get('alt', '')}](images/{filename})\n\n"
                    f"*{spec.get('caption', '')}*"
                )
                markdown = markdown.replace(placeholder, image_markdown)
                generated_images.append({
                    "path": str(output_path),
                    "filename": filename,
                    "caption": spec.get("caption", ""),
                    "alt": spec.get("alt", ""),
                })
            except Exception as exc:
                error_message = str(exc)
                errors.append(f"{safe_stem}: {error_message}")
                markdown = markdown.replace(
                    placeholder,
                    f"> Image generation failed for {safe_stem}: {error_message}",
                )
    elif specs and not os.getenv("POLLINATIONS_API_KEY"):
        errors.append("POLLINATIONS_API_KEY is missing.")
    # Remove any unfilled placeholders so they do not appear in the blog.
    markdown = re.sub(r"\[\[IMAGE_\d+\]\]", "", markdown)
    plan = state.get("plan")
    title = plan.blog_title if plan else "blog"
    slug = _slugify(title)
    output_file = BASE_DIR / f"{slug}.md"
    output_file.write_text(markdown, encoding="utf-8")
    if generated_images and not errors:
        image_status = f"Successfully generated {len(generated_images)} AI image(s)."
    elif generated_images and errors:
        image_status = (
            f"Generated {len(generated_images)} image(s); "
            f"{len(errors)} image(s) failed. " + " | ".join(errors)
        )
    elif errors:
        image_status = "Image generation did not complete: " + " | ".join(errors)
    else:
        image_status = state.get("image_status", "No images were generated.")
    return {
        "final": markdown,
        "generated_images": generated_images,
        "image_status": image_status,
    }

# ============================================================
# BUILD LANGGRAPH

# ============================================================
graph = StateGraph(BlogState)
graph.add_node("router", router_node)
graph.add_node("research", research_node)
graph.add_node("planner", planner_node)
graph.add_node("writer", writer_node)
graph.add_node("merge", merge_node)
graph.add_node("image_plan", image_plan_node)
graph.add_node("place_images", place_images_node)
graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    route_after_router,
    {
        "research": "research",
        "planner": "planner",
    },
)
graph.add_edge("research", "planner")

def fanout_workers(state: BlogState):
    """Dispatch one writer task per planned blog section in parallel."""
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Plan missing: planner_node did not return a BlogPlan.")
    return [
        Send(
            "writer",
            {
                "task": task.model_dump(),
                "plan": plan.model_dump(),
                "evidence": [
                    item.model_dump() if isinstance(item, EvidenceItem) else item
                    for item in state.get("evidence", [])
                ],
                "topic": state.get("topic", ""),
            },
        )
        for task in plan.tasks
    ]
graph.add_conditional_edges(
    "planner",
    fanout_workers,
    ["writer"],
)
graph.add_edge("writer", "merge")
graph.add_edge("merge", "image_plan")
graph.add_edge("image_plan", "place_images")
graph.add_edge("place_images", END)
app = graph.compile()
