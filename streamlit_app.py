from io import BytesIO
import math

import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from qed_diagrams import (
    DiagramRequest,
    draw_diagram,
    export_figure,
    generate_diagrams,
    loop_order_from_counts,
    validate_request,
    vertex_count_from_loops,
)


@st.cache_data(max_entries=16, show_spinner="Enumerating QED diagrams…")
def cached_generate_diagrams(request: DiagramRequest, limit: int):
    return generate_diagrams(request, limit)


def make_contact_sheets(
    png_images: list[bytes], mode: str, diagrams_per_sheet: int
) -> list[bytes]:
    """Arrange diagrams into one auto-scaled PNG or several PNG pages."""
    groups = (
        [png_images]
        if mode == "Fit into one image"
        else [png_images[i:i + diagrams_per_sheet] for i in range(0, len(png_images), diagrams_per_sheet)]
    )
    outputs: list[bytes] = []
    for group in groups:
        count = len(group)
        columns = max(1, math.ceil(math.sqrt(count * 0.625)))
        if mode == "Split into multiple images":
            columns = min(4, columns)
        rows = math.ceil(count / columns)
        cell_width, cell_height = 480, 300
        if mode == "Fit into one image":
            scale = min(1.0, 8000 / (columns * cell_width), 8000 / (rows * cell_height))
            cell_width = max(120, round(cell_width * scale))
            cell_height = max(75, round(cell_height * scale))
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#080d18")
        for index, data in enumerate(group):
            with Image.open(BytesIO(data)) as source:
                diagram = source.convert("RGB")
            diagram.thumbnail((cell_width - 12, cell_height - 12), Image.Resampling.LANCZOS)
            left = (index % columns) * cell_width + (cell_width - diagram.width) // 2
            top = (index // columns) * cell_height + (cell_height - diagram.height) // 2
            sheet.paste(diagram, (left, top))
        output = BytesIO()
        sheet.save(output, format="PNG", optimize=True)
        outputs.append(output.getvalue())
    return outputs


st.set_page_config(page_title="QED diagram studio", page_icon=":material/hub:", layout="wide")
st.title("QED diagram studio")
st.caption("Generate only Feynman diagrams that satisfy QED rules, the 1PI condition, and Furry's theorem.")

with st.sidebar:
    st.header("Input")
    order_mode = st.segmented_control(
        "Specify perturbative order by",
        ["Vertex count", "Loop order"],
        default="Loop order",
        key="order_mode",
    )
    with st.form("diagram_request"):
        st.subheader("Incoming external lines")
        electron_in = st.number_input("Electron e⁻", min_value=0, max_value=8, value=1, step=1, key="electron_in")
        positron_in = st.number_input("Positron e⁺", min_value=0, max_value=8, value=0, step=1, key="positron_in")
        photon_in = st.number_input("Photon γ", min_value=0, max_value=8, value=0, step=1, key="photon_in")
        st.subheader("Outgoing external lines")
        electron_out = st.number_input("Electron e⁻", min_value=0, max_value=8, value=1, step=1, key="electron_out")
        positron_out = st.number_input("Positron e⁺", min_value=0, max_value=8, value=0, step=1, key="positron_out")
        photon_out = st.number_input("Photon γ", min_value=0, max_value=8, value=0, step=1, key="photon_out")
        if order_mode == "Vertex count":
            order_value = st.number_input("Vertex count V", min_value=1, max_value=12, value=5, step=1)
        else:
            order_value = st.number_input("Loop order L", min_value=0, max_value=5, value=1, step=1)
        limit = st.number_input(
            "Maximum number of diagrams",
            min_value=1,
            max_value=1000,
            value=8,
            step=1,
            help="If more diagrams satisfy the conditions, only this many will be displayed.",
        )
        submitted = st.form_submit_button("Generate diagrams", type="primary", icon=":material/auto_awesome:")
    st.caption("Fermion arrows point in opposite directions for electrons and positrons.")

if "request" not in st.session_state:
    st.session_state.request = DiagramRequest(1, 0, 1, 0, 2, 0, 0, 1)
    st.session_state.limit = 8
if submitted:
    st.session_state.pop("contact_sheet_result", None)
    st.session_state.pop("contact_sheet_key", None)
    st.session_state.diagram_page = 1
    external_lines = int(electron_in + positron_in + photon_in + electron_out + positron_out + photon_out)
    if order_mode == "Vertex count":
        vertices = int(order_value)
        derived_loops = loop_order_from_counts(external_lines, vertices)
        loops = derived_loops if derived_loops is not None else -1
    else:
        loops = int(order_value)
        vertices = vertex_count_from_loops(external_lines, loops)
    st.session_state.request = DiagramRequest(
        int(electron_in), int(photon_in), int(electron_out), int(photon_out), int(vertices),
        int(positron_in), int(positron_out), int(loops),
    )
    st.session_state.limit = int(limit)

req = st.session_state.request
valid, message, internal_fermions, internal_photons = validate_request(req)

metrics = st.columns(4)
metrics[0].metric("Vertices", req.vertices)
metrics[1].metric("Loop order", req.loops)
metrics[2].metric("Internal fermion lines", internal_fermions if valid else "—")
metrics[3].metric("Internal photon lines", internal_photons if valid else "—")
st.caption(f"For E={req.electron_in + req.positron_in + req.photon_in + req.electron_out + req.positron_out + req.photon_out} external lines, V={req.vertices} and L={req.loops} are linked automatically.")

if not valid:
    st.error(message, icon=":material/error:")
    st.stop()

diagrams = cached_generate_diagrams(req, st.session_state.limit)
if not diagrams:
    st.warning("No diagram satisfies the QED rules, the 1PI condition, and Furry's theorem. Tree diagrams are normally excluded because cutting one internal line disconnects them.", icon=":material/warning:")
    st.stop()

st.success(f"Generated {len(diagrams)} distinct representative diagrams.", icon=":material/check_circle:")
st.caption("Blue arrows are fermion lines, yellow waves are photon lines, and red dots are QED vertices. Exact graph isomorphism removes duplicates. pyfeyn2 is used for structure and bend metadata.")

page_size = 8
total_pages = math.ceil(len(diagrams) / page_size)
with st.container(border=True):
    st.subheader("Individual diagrams")
    st.session_state.setdefault("diagram_page", 1)
    page_number = int(st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        step=1,
        key="diagram_page",
        help=f"Displays {page_size} diagrams per page. There are {total_pages} pages.",
    ))
    st.caption(f"Showing diagrams {(page_number - 1) * page_size + 1}–{min(page_number * page_size, len(diagrams))} of {len(diagrams)}")

page_start = (page_number - 1) * page_size
page_end = min(page_start + page_size, len(diagrams))
for row_start in range(page_start, page_end, 2):
    columns = st.columns(2)
    for column, index in zip(columns, range(row_start, min(row_start + 2, page_end))):
        diagram = diagrams[index]
        with column.container(border=True):
            fig = draw_diagram(diagram, f"Diagram {index + 1}")
            st.pyplot(fig, width="stretch")
            png = export_figure(fig, "png")
            svg = export_figure(fig, "svg")
            plt.close(fig)
            with st.container(horizontal=True):
                st.download_button("PNG", png, f"qed_diagram_{index + 1}.png", "image/png", icon=":material/download:", key=f"png_{index}")
                st.download_button("SVG", svg, f"qed_diagram_{index + 1}.svg", "image/svg+xml", icon=":material/download:", key=f"svg_{index}")

with st.container(border=True):
    st.subheader("Contact sheet")
    contact_sheet_mode = st.segmented_control(
        "If the image becomes too large",
        ["Fit into one image", "Split into multiple images"],
        default="Fit into one image",
        key="contact_sheet_mode",
    )
    diagrams_per_sheet = 24
    if contact_sheet_mode == "Split into multiple images":
        diagrams_per_sheet = int(st.number_input(
            "Diagrams per image",
            min_value=1,
            max_value=100,
            value=24,
            step=1,
            key="diagrams_per_sheet",
        ))
    create_contact_sheet = st.button(
        "Create contact sheet",
        type="primary",
        icon=":material/grid_view:",
        key="create_contact_sheet",
    )
    st.caption("The contact sheet is generated only when you press this button.")

contact_key = (req, len(diagrams), contact_sheet_mode, diagrams_per_sheet)
if create_contact_sheet:
    all_pngs: list[bytes] = []
    progress = st.progress(0, text="Rendering diagrams for the contact sheet…")
    for index, diagram in enumerate(diagrams):
        fig = draw_diagram(diagram, f"Diagram {index + 1}")
        all_pngs.append(export_figure(fig, "png"))
        plt.close(fig)
        progress.progress((index + 1) / len(diagrams), text=f"Rendered {index + 1}/{len(diagrams)} diagrams")
    contact_sheets = make_contact_sheets(all_pngs, contact_sheet_mode, diagrams_per_sheet)
    progress.empty()
    st.session_state.contact_sheet_result = contact_sheets
    st.session_state.contact_sheet_key = contact_key
    st.success("Contact sheet created.", icon=":material/check_circle:")

if st.session_state.get("contact_sheet_key") == contact_key:
    contact_sheets = st.session_state.contact_sheet_result
    st.subheader("Download contact sheets")
    st.caption(
        "When fitted into one image, it is automatically scaled to a maximum of 8000 px."
        if contact_sheet_mode == "Fit into one image"
        else f"Split into {len(contact_sheets)} images with up to {diagrams_per_sheet} diagrams each."
    )
    for page, sheet in enumerate(contact_sheets, 1):
        filename = "qed_diagrams_all.png" if len(contact_sheets) == 1 else f"qed_diagrams_page_{page}.png"
        st.download_button(
            f"Download contact sheet {page}" if len(contact_sheets) > 1 else "Download contact sheet",
            sheet,
            filename,
            "image/png",
            icon=":material/download:",
            key=f"contact_sheet_{page}",
        )
        preview = st.expander(f"Preview contact sheet {page}", icon=":material/image:", on_change="rerun")
        if preview.open:
            with preview:
                st.image(sheet, width="stretch")

with st.expander("Selection rules", icon=":material/function:"):
    st.latex(r"I_\gamma=(V-E_\gamma)/2,\quad I_e=(2V-E_e)/2,\quad L=I_e+I_\gamma-V+1")
    st.write("Each QED vertex connects two fermion lines and one photon line. Diagrams must conserve charge, satisfy the equation above, and form a connected graph.")
    st.write("A diagram is accepted as 1PI only if it remains connected after any single internal propagator is removed.")
    st.write("By Furry's theorem, closed fermion loops with an odd number of photon vertices have zero amplitude and are excluded.")
