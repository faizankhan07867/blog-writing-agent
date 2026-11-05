from __future__ import annotations

import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

import streamlit as st

from bwa_backend import app


# ============================
# PAGE CONFIGURATION
# ============================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"

st.set_page_config(
    page_title="AI Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
)

st.title("✍️ AI Blog Writing Agent")
st.caption("Research, plan, write and generate professional blogs using AI.")


# ============================
# SESSION STATE
# ============================

if "blog_output" not in st.session_state:
    st.session_state.blog_output = None

if "event_logs" not in st.session_state:
    st.session_state.event_logs = []


# ============================
# HELPERS
# ============================

IMAGE_MARKDOWN_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def get_generated_image_paths(output: dict) -> list[Path]:
    """Return only image files recorded by the backend for this blog run."""
    paths: list[Path] = []

    for item in output.get("generated_images", []) or []:
        raw_path = item.get("path") if isinstance(item, dict) else None
        if not raw_path:
            continue

        path = Path(raw_path)
        if not path.is_absolute():
            path = BASE_DIR / path

        if path.is_file() and path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp"
        }:
            paths.append(path)

    # Remove duplicate paths while preserving order.
    return list(dict.fromkeys(paths))


def resolve_markdown_image(target: str, generated_paths: list[Path]) -> Path | None:
    """Resolve a local Markdown image target to a real file on disk."""
    target = unquote(target.strip().strip("'").strip('"'))

    if target.startswith(("http://", "https://")):
        return None

    candidate = Path(target)
    candidates = []

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                BASE_DIR / candidate,
                BASE_DIR / "images" / candidate.name,
            ]
        )

    generated_by_name = {path.name: path for path in generated_paths}
    if candidate.name in generated_by_name:
        candidates.insert(0, generated_by_name[candidate.name])

    for path in candidates:
        if path.is_file():
            return path

    return None


def render_blog_markdown(markdown: str, generated_paths: list[Path]) -> None:
    """Render Markdown text and embed locally generated images in the preview."""
    cursor = 0

    for match in IMAGE_MARKDOWN_RE.finditer(markdown):
        text_before = markdown[cursor:match.start()]
        if text_before.strip():
            st.markdown(text_before)

        alt_text = match.group(1).strip()
        target = match.group(2).strip()

        if target.startswith(("http://", "https://")):
            st.image(target, caption=alt_text or None, use_container_width=True)
        else:
            image_path = resolve_markdown_image(target, generated_paths)
            if image_path:
                st.image(
                    str(image_path),
                    caption=alt_text or image_path.name,
                    use_container_width=True,
                )
            else:
                st.warning(f"Image file could not be found: {target}")

        cursor = match.end()

    remaining = markdown[cursor:]
    if remaining.strip():
        st.markdown(remaining)


def make_blog_zip(slug: str, markdown: str, image_paths: list[Path]) -> bytes:
    """Create a ZIP containing the blog Markdown and this run's generated images."""
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{slug}.md", markdown)

        for image_path in image_paths:
            if image_path.is_file():
                archive.write(image_path, arcname=f"images/{image_path.name}")

    return buffer.getvalue()


# ============================
# SIDEBAR
# ============================

with st.sidebar:
    st.header("Generate New Blog")

    topic = st.text_area(
        "Enter Blog Topic",
        placeholder="Example: Python Full Stack Development",
        height=130,
    )

    as_of = st.date_input("As-of Date", value=date.today())

    generate_images = st.checkbox(
        "Generate AI Images",
        value=False,
        help="Turn this on to ask the backend to generate real images using Pollinations.",
    )

    image_count = st.slider(
        "Number of images",
        min_value=1,
        max_value=3,
        value=2,
        disabled=not generate_images,
    )

    run_button = st.button(
        "🚀 Generate Blog",
        type="primary",
        use_container_width=True,
    )


# ============================
# TABS
# ============================

tab_blog, tab_plan, tab_evidence, tab_images, tab_logs = st.tabs(
    [
        "📝 Blog Preview",
        "🧩 Plan",
        "🔎 Research",
        "🖼️ Images",
        "🧾 Logs",
    ]
)


# ============================
# GENERATE BLOG
# ============================

if run_button:
    if not topic.strip():
        st.warning("Please enter a blog topic.")
        st.stop()

    initial_state = {
        "topic": topic.strip(),
        "as_of": as_of.isoformat(),
        "mode": "closed_book",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "merged_md": "",
        "markdown_with_placeholders": "",
        "image_specs": [],
        "generated_images": [],
        "image_status": "",
        "image_count": image_count,
        "final": "",
        "generate_images": generate_images,
    }

    progress = st.status("Generating your blog...", expanded=True)
    progress_area = st.empty()

    try:
        final_state = None

        for state in app.stream(
            initial_state,
            config={"max_concurrency": 1},
            stream_mode="values",
        ):
            final_state = state
            plan = state.get("plan")

            if plan:
                progress_area.write(f"Plan: {plan.blog_title}")
                progress_area.write(
                    f"Sections completed: {len(state.get('sections', []))}"
                )
            else:
                progress_area.write("Analyzing topic and preparing research...")

            if state.get("image_status"):
                progress_area.write(state["image_status"])

        if not final_state or not final_state.get("final"):
            raise RuntimeError("No final blog was returned by LangGraph.")

        st.session_state.blog_output = final_state
        st.session_state.event_logs.append("Blog generated successfully.")

        image_status = final_state.get("image_status")
        if image_status:
            st.session_state.event_logs.append(str(image_status))

        generated_count = len(final_state.get("generated_images", []) or [])
        if generate_images and generated_count == 0:
            st.session_state.event_logs.append(
                "Image generation was enabled, but no image file was returned. "
                "Check the Images tab for the backend error/status."
            )

        progress.update(
            label="Blog generated successfully!",
            state="complete",
            expanded=False,
        )

    except Exception as exc:
        progress.update(
            label="Blog generation failed",
            state="error",
            expanded=True,
        )
        st.session_state.event_logs.append(f"ERROR: {exc}")
        st.exception(exc)


# ============================
# DISPLAY RESULTS
# ============================

output = st.session_state.blog_output

if output:
    plan = output.get("plan")
    final_markdown = output.get("final", "")
    title = plan.blog_title if plan else "Generated Blog"

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "blog"
    generated_image_paths = get_generated_image_paths(output)

    # ------------------------
    # BLOG PREVIEW
    # ------------------------

    with tab_blog:
        st.subheader(title)

        render_blog_markdown(final_markdown, generated_image_paths)

        st.divider()

        st.download_button(
            "⬇️ Download Markdown",
            data=final_markdown.encode("utf-8"),
            file_name=f"{slug}.md",
            mime="text/markdown",
        )

        if generated_image_paths:
            st.download_button(
                "📦 Download Blog + Generated Images",
                data=make_blog_zip(slug, final_markdown, generated_image_paths),
                file_name=f"{slug}-bundle.zip",
                mime="application/zip",
            )

    # ------------------------
    # PLAN
    # ------------------------

    with tab_plan:
        if plan:
            st.subheader("Blog Planning")
            st.write("**Title:**", plan.blog_title)
            st.write("**Audience:**", plan.audience)
            st.write("**Tone:**", plan.tone)
            st.write("**Blog Type:**", plan.blog_kind)

            st.dataframe(
                [
                    {
                        "ID": task.id,
                        "Section": task.title,
                        "Target Words": task.target_words,
                        "Research": task.requires_research,
                        "Citations": task.requires_citations,
                        "Code": task.requires_code,
                    }
                    for task in plan.tasks
                ],
                use_container_width=True,
                hide_index=True,
            )

            if plan.constraints:
                st.write("**Constraints:**", ", ".join(plan.constraints))
        else:
            st.info("No plan found.")

    # ------------------------
    # RESEARCH
    # ------------------------

    with tab_evidence:
        evidence = output.get("evidence", [])

        if evidence:
            for item in evidence:
                if isinstance(item, dict):
                    item_title = item.get("title", "Research source")
                    item_url = item.get("url", "")
                    snippet = item.get("snippet", "")
                else:
                    item_title = item.title
                    item_url = item.url
                    snippet = item.snippet

                if item_url:
                    st.markdown(f"- [{item_title}]({item_url})")
                else:
                    st.markdown(f"- {item_title}")

                if snippet:
                    st.caption(snippet[:500])
        else:
            research_status = output.get("research_status", "")
            st.info(
                research_status
                or "No research results available. Configure TAVILY_API_KEY "
                   "in your .env file or check the DuckDuckGo fallback."
            )

    # ------------------------
    # IMAGES
    # ------------------------

    with tab_images:
        image_status = output.get("image_status", "")
        image_specs = output.get("image_specs", [])

        st.subheader("Generated AI Images")

        if image_status:
            st.info(image_status)

        if generated_image_paths:
            st.success(f"{len(generated_image_paths)} real image(s) generated.")

            columns = st.columns(2)
            for index, image_path in enumerate(generated_image_paths):
                with columns[index % 2]:
                    st.image(
                        str(image_path),
                        caption=image_path.name,
                        use_container_width=True,
                    )
        elif not output.get("generate_images", False):
            st.warning(
                "Image generation was not enabled for this blog. "
                "Turn on 'Generate AI Images' in the sidebar and generate again."
            )
        else:
            st.error(
                "No actual image file was generated. Read the image status above "
                "and verify POLLINATIONS_API_KEY in your .env file."
            )

        if image_specs:
            with st.expander("View image prompts (debug details)"):
                for index, spec in enumerate(image_specs, start=1):
                    st.markdown(f"**Image {index}: {spec.get('caption', 'Image')}**")
                    st.caption(spec.get("alt", ""))
                    st.code(spec.get("prompt", ""), language="text")

    # ------------------------
    # LOGS
    # ------------------------

    with tab_logs:
        st.subheader("Execution Logs")

        if st.session_state.event_logs:
            st.text_area(
                "Event Logs",
                value="\n".join(st.session_state.event_logs),
                height=300,
            )
        else:
            st.info("No execution logs yet.")

else:
    with tab_blog:
        st.info("Enter a topic and click Generate Blog.")
