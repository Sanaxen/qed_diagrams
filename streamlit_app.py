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


@st.cache_data(max_entries=16, show_spinner="QED図を列挙しています…")
def cached_generate_diagrams(request: DiagramRequest, limit: int):
    return generate_diagrams(request, limit)


def make_contact_sheets(
    png_images: list[bytes], mode: str, diagrams_per_sheet: int
) -> list[bytes]:
    """Arrange diagrams into one auto-scaled PNG or several PNG pages."""
    groups = (
        [png_images]
        if mode == "1枚に収める"
        else [png_images[i:i + diagrams_per_sheet] for i in range(0, len(png_images), diagrams_per_sheet)]
    )
    outputs: list[bytes] = []
    for group in groups:
        count = len(group)
        columns = max(1, math.ceil(math.sqrt(count * 0.625)))
        if mode == "複数画像に分割":
            columns = min(4, columns)
        rows = math.ceil(count / columns)
        cell_width, cell_height = 480, 300
        if mode == "1枚に収める":
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
st.caption("外線と頂点数から、QED規則・1PI条件・Furryの定理を満たすファインマン図だけを生成します。")

with st.sidebar:
    st.header("入力")
    order_mode = st.segmented_control(
        "摂動次数の指定方法",
        ["頂点数", "ループ次数"],
        default="ループ次数",
        key="order_mode",
    )
    with st.form("diagram_request"):
        st.subheader("入射外線")
        electron_in = st.number_input("電子 e⁻", min_value=0, max_value=8, value=1, step=1, key="electron_in")
        positron_in = st.number_input("陽電子 e⁺", min_value=0, max_value=8, value=0, step=1, key="positron_in")
        photon_in = st.number_input("光子 γ", min_value=0, max_value=8, value=0, step=1, key="photon_in")
        st.subheader("出射外線")
        electron_out = st.number_input("電子 e⁻", min_value=0, max_value=8, value=1, step=1, key="electron_out")
        positron_out = st.number_input("陽電子 e⁺", min_value=0, max_value=8, value=0, step=1, key="positron_out")
        photon_out = st.number_input("光子 γ", min_value=0, max_value=8, value=0, step=1, key="photon_out")
        if order_mode == "頂点数":
            order_value = st.number_input("頂点数 V", min_value=1, max_value=12, value=5, step=1)
        else:
            order_value = st.number_input("ループ次数 L", min_value=0, max_value=5, value=1, step=1)
        limit = st.number_input(
            "表示する図の最大数",
            min_value=1,
            max_value=1000,
            value=8,
            step=1,
            help="条件を満たす図が多い場合も、ここで指定した枚数まで表示します。",
        )
        submitted = st.form_submit_button("図を生成", type="primary", icon=":material/auto_awesome:")
    st.caption("フェルミオンの矢印は、電子と陽電子で逆向きになります。")

if "request" not in st.session_state:
    st.session_state.request = DiagramRequest(1, 0, 1, 0, 2, 0, 0, 1)
    st.session_state.limit = 8
if submitted:
    st.session_state.pop("contact_sheet_result", None)
    st.session_state.pop("contact_sheet_key", None)
    st.session_state.diagram_page = 1
    external_lines = int(electron_in + positron_in + photon_in + electron_out + positron_out + photon_out)
    if order_mode == "頂点数":
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
metrics[0].metric("頂点", req.vertices)
metrics[1].metric("ループ次数", req.loops)
metrics[2].metric("内部電子線", internal_fermions if valid else "—")
metrics[3].metric("内部光子線", internal_photons if valid else "—")
st.caption(f"外線総数 E={req.electron_in + req.positron_in + req.photon_in + req.electron_out + req.positron_out + req.photon_out} から、V={req.vertices}、L={req.loops} と自動対応付けしています。")

if not valid:
    st.error(message, icon=":material/error:")
    st.stop()

diagrams = cached_generate_diagrams(req, st.session_state.limit)
if not diagrams:
    st.warning("QED規則、1PI条件、Furryの定理をすべて満たす図がありません。ツリー図は内部線を1本切ると分離するため、通常は1PI候補から除外されます。", icon=":material/warning:")
    st.stop()

st.success(f"{len(diagrams)} 個の異なる代表図を生成しました。", icon=":material/check_circle:")
st.caption("青い矢印はフェルミオン線、黄色い波線は光子線、赤い点はQED頂点です。厳密なグラフ同型判定で重複を除外しています。構造化とベンド判定にはpyfeyn2を使用しています。")

page_size = 8
total_pages = math.ceil(len(diagrams) / page_size)
with st.container(border=True):
    st.subheader("個別図の表示")
    st.session_state.setdefault("diagram_page", 1)
    page_number = int(st.number_input(
        "表示ページ",
        min_value=1,
        max_value=total_pages,
        step=1,
        key="diagram_page",
        help=f"1ページあたり{page_size}図を表示します。全{total_pages}ページです。",
    ))
    st.caption(f"全{len(diagrams)}図中、{(page_number - 1) * page_size + 1}～{min(page_number * page_size, len(diagrams))}図を表示")

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
    st.subheader("まとめ画像")
    contact_sheet_mode = st.segmented_control(
        "画像が大きくなる場合",
        ["1枚に収める", "複数画像に分割"],
        default="1枚に収める",
        key="contact_sheet_mode",
    )
    diagrams_per_sheet = 24
    if contact_sheet_mode == "複数画像に分割":
        diagrams_per_sheet = int(st.number_input(
            "1画像あたりの図数",
            min_value=1,
            max_value=100,
            value=24,
            step=1,
            key="diagrams_per_sheet",
        ))
    create_contact_sheet = st.button(
        "まとめ画像を作成",
        type="primary",
        icon=":material/grid_view:",
        key="create_contact_sheet",
    )
    st.caption("このボタンを押した時だけ、全図のまとめ画像を作成します。")

contact_key = (req, len(diagrams), contact_sheet_mode, diagrams_per_sheet)
if create_contact_sheet:
    all_pngs: list[bytes] = []
    progress = st.progress(0, text="まとめ画像用の図を描画しています…")
    for index, diagram in enumerate(diagrams):
        fig = draw_diagram(diagram, f"Diagram {index + 1}")
        all_pngs.append(export_figure(fig, "png"))
        plt.close(fig)
        progress.progress((index + 1) / len(diagrams), text=f"{index + 1}/{len(diagrams)}図を描画しました")
    contact_sheets = make_contact_sheets(all_pngs, contact_sheet_mode, diagrams_per_sheet)
    progress.empty()
    st.session_state.contact_sheet_result = contact_sheets
    st.session_state.contact_sheet_key = contact_key
    st.success("まとめ画像を作成しました。", icon=":material/check_circle:")

if st.session_state.get("contact_sheet_key") == contact_key:
    contact_sheets = st.session_state.contact_sheet_result
    st.subheader("まとめ画像のダウンロード")
    st.caption(
        "1枚に収める場合は最大8000px以内へ自動縮小します。"
        if contact_sheet_mode == "1枚に収める"
        else f"{diagrams_per_sheet}図ごとに{len(contact_sheets)}枚へ分割しました。"
    )
    for page, sheet in enumerate(contact_sheets, 1):
        filename = "qed_diagrams_all.png" if len(contact_sheets) == 1 else f"qed_diagrams_page_{page}.png"
        st.download_button(
            f"まとめ画像 {page} をダウンロード" if len(contact_sheets) > 1 else "まとめ画像をダウンロード",
            sheet,
            filename,
            "image/png",
            icon=":material/download:",
            key=f"contact_sheet_{page}",
        )
        preview = st.expander(f"まとめ画像 {page} のプレビュー", icon=":material/image:", on_change="rerun")
        if preview.open:
            with preview:
                st.image(sheet, width="stretch")

with st.expander("判定ルール", icon=":material/function:"):
    st.latex(r"I_\gamma=(V-E_\gamma)/2,\quad I_e=(2V-E_e)/2,\quad L=I_e+I_\gamma-V+1")
    st.write("各QED頂点にはフェルミオン線が2本、光子線が1本接続します。電荷保存と上式を満たし、生成されたグラフが連結になるものを表示します。")
    st.write("さらに、どの内部プロパゲータを1本除いても連結性が保たれる図だけを1PIとして採用します。")
    st.write("Furryの定理により、奇数個の光子頂点を持つ閉フェルミオンループは振幅が0になるため除外します。")
