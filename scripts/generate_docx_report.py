import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def create_report():
    doc = Document()
    
    # ---------------------------------------------------------
    # Page Setup (Standard Academic Margins)
    # Top: 2.0 cm (0.79 in), Bottom: 2.0 cm (0.79 in), Left: 3.0 cm (1.18 in), Right: 2.0 cm (0.79 in)
    # ---------------------------------------------------------
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(1.18)
        section.right_margin = Inches(0.8)
        
    # Styles Setup (Standard Academic Font: Times New Roman, 13pt)
    normal_style = doc.styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Times New Roman'
    normal_font.size = Pt(13)
    normal_font.color.rgb = RGBColor(0x00, 0x00, 0x00) # Pure Black
    normal_style.paragraph_format.line_spacing = 1.3
    normal_style.paragraph_format.space_after = Pt(6)
    normal_style.paragraph_format.space_before = Pt(0)

    COLOR_BLACK = RGBColor(0x00, 0x00, 0x00)
    HEX_TABLE_HEADER_BG = "E6E6E6"  # Light Neutral Gray for academic table header
    HEX_TABLE_ALT_BG = "F9F9F9"     # Very soft tint for alt rows
    HEX_BORDER = "BFBFBF"           # Neutral Border Gray
    
    # Helper Functions for Academic Formatting
    def set_cell_background(cell, fill_hex):
        tcPr = cell._element.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        tcPr.append(shd)

    def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
        tcPr = cell._element.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for margin, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
            node = OxmlElement(f'w:{margin}')
            node.set(qn('w:w'), str(val))
            node.set(qn('w:type'), 'dxa')
            tcMar.append(node)
        tcPr.append(tcMar)

    def set_table_borders(table, color="A6A6A6", sz="4", val="single"):
        tblPr = table._element.xpath('w:tblPr')
        if tblPr:
            borders = parse_xml(f'''
                <w:tblBorders {nsdecls("w")}>
                    <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                    <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                    <w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                    <w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                    <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                    <w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                </w:tblBorders>
            ''')
            tblPr[0].append(borders)

    def add_academic_title(main_title, sub_title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(main_title)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(18)
        r.font.bold = True
        r.font.color.rgb = COLOR_BLACK
        
        p_sub = doc.add_paragraph()
        p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_sub.paragraph_format.space_before = Pt(0)
        p_sub.paragraph_format.space_after = Pt(24)
        r_sub = p_sub.add_run(sub_title)
        r_sub.font.name = 'Times New Roman'
        r_sub.font.size = Pt(14)
        r_sub.font.bold = True
        r_sub.font.color.rgb = COLOR_BLACK

    def add_chapter_heading(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(15)
        r.font.bold = True
        r.font.color.rgb = COLOR_BLACK
        return p

    def add_section_heading(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(13.5)
        r.font.bold = True
        r.font.color.rgb = COLOR_BLACK
        return p

    def add_subsection_heading(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = COLOR_BLACK
        return p

    def add_body_paragraph(text, indent=True):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.3
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.4)
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(13)
        r.font.color.rgb = COLOR_BLACK
        return p

    def add_list_item(prefix_text, body_text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.line_spacing = 1.3
        
        r_pre = p.add_run(prefix_text + " ")
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(13)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_BLACK
        
        r_body = p.add_run(body_text)
        r_body.font.name = 'Times New Roman'
        r_body.font.size = Pt(13)
        r_body.font.color.rgb = COLOR_BLACK

    def add_image_with_caption(img_path, caption_text, figure_num="", width_inches=5.8):
        if os.path.exists(img_path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run()
            run.add_picture(img_path, width=Inches(width_inches))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(12)
            
            r_num = p_cap.add_run(f"Hình {figure_num}: " if figure_num else "")
            r_num.font.name = 'Times New Roman'
            r_num.font.size = Pt(11)
            r_num.font.bold = True
            
            r_cap = p_cap.add_run(caption_text)
            r_cap.font.name = 'Times New Roman'
            r_cap.font.size = Pt(11)
            r_cap.font.italic = True
            r_cap.font.color.rgb = COLOR_BLACK

    # ---------------------------------------------------------
    # TRANG BÌA & THÔNG TIN DỰ ÁN
    # ---------------------------------------------------------
    add_academic_title(
        "BÁO CÁO KĨ THUẬT DỰ ÁN TỐT NGHIỆP",
        "NGHIÊN CỨU VÀ XÂY DỰNG HỆ THỐNG PHÂN LOẠI CẢM XÚC BÌNH LUẬN TIẾNG VIỆT SỬ DỤNG MÔ HÌNH PHOBERT-AMP"
    )

    # Info Table (Standard Academic Border & Clean Background)
    info_tbl = doc.add_table(rows=5, cols=2)
    info_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        ("Tên đề tài dự án:", "Hệ thống phân loại cảm xúc bình luận thương mại điện tử tiếng Việt"),
        ("Mô hình đề xuất:", "PhoBERT-AMP (Adaptive Metadata-aware PhoBERT)"),
        ("Tập dữ liệu kiểm thử:", "15.378 bình luận thương mại điện tử (Shopee, Tiki, Lazada)"),
        ("Hiệu năng đạt được:", "Accuracy: 99,61% | Macro F1-Score: 99,56%"),
        ("Đơn vị thực hiện:", "Đồ án tốt nghiệp - Ngành Công nghệ Thông tin")
    ]
    for row_idx, (label, val) in enumerate(info_data):
        c0 = info_tbl.cell(row_idx, 0)
        c1 = info_tbl.cell(row_idx, 1)
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        set_cell_background(c0, HEX_TABLE_ALT_BG)
        set_cell_background(c1, HEX_TABLE_ALT_BG)
        set_cell_margins(c0, top=80, bottom=80, left=100, right=100)
        set_cell_margins(c1, top=80, bottom=80, left=100, right=100)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(label)
        r0.font.name = 'Times New Roman'
        r0.font.bold = True
        r0.font.size = Pt(12)
        
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(val)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(12)

    set_table_borders(info_tbl, color=HEX_BORDER, sz="6")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ---------------------------------------------------------
    # CHƯƠNG 1: TỔNG QUAN VỀ DỰ ÁN VÀ MỤC TIÊU ĐỀ TÀI
    # ---------------------------------------------------------
    add_chapter_heading("CHƯƠNG 1: TỔNG QUAN VỀ DỰ ÁN VÀ MỤC TIÊU ĐỀ TÀI")

    add_section_heading("1.1. Bối cảnh thực tiễn và tính cấp thiết của đề tài")
    add_body_paragraph(
        "Sự phát triển mạnh mẽ của thị trường thương mại điện tử tại Việt Nam trong những năm gần đây đã tạo ra một lượng dữ liệu phản hồi (feedback) "
        "và bình luận (reviews) khổng lồ từ phía người tiêu dùng trên các sàn giao dịch lớn như Shopee, Lazada, Tiki... Phân tích cảm xúc "
        "(Sentiment Analysis) từ văn bản tiếng Việt đóng vai trò đặc biệt quan trọng giúp các doanh nghiệp tự động hóa quá trình theo dõi mức độ "
        "hài lòng của khách hàng, kịp thời phát hiện sản phẩm lỗi và xử lý các khủng hoảng truyền thông."
    )
    add_body_paragraph(
        "Tuy nhiên, xử lý ngôn ngữ tự nhiên (NLP) áp dụng cho bình luận thương mại điện tử tiếng Việt gặp phải nhiều thách thức lớn như:"
    )
    add_list_item("a)", "Đặc thù ngôn ngữ tự do: Người dùng thường xuyên sử dụng từ lóng (teencode), từ viết tắt không chuẩn hóa ('sp', 'ko', 'dmel'), ký tự emoji/icon, cũng như mắc lỗi chính tả hoặc ngắt câu không rõ ràng.")
    add_list_item("b)", "Sự ngắn gọn và thiếu thông tin văn bản: Nhiều bình luận chỉ vỏn vẹn một vài từ như 'Ok', 'Đã nhận hàng', 'Cũng được'... khiến các mô hình NLP thuần túy dựa vào ngữ nghĩa văn bản rất khó đưa ra nhãn cảm xúc chính xác.")
    add_list_item("c)", "Hiện tượng mỉa mai hoặc bất đồng thông tin: Có những bình luận có nội dung mang tính khen ngợi ẩn ý hoặc mỉa mai nhưng điểm số đánh giá (rating) lại thể hiện sự không hài lòng.")

    add_section_heading("1.2. Mục tiêu nghiên cứu của dự án")
    add_body_paragraph(
        "Đề tài hướng tới nghiên cứu, xây dựng và triển khai một hệ thống học sâu tiên tiến mang tên PhoBERT-AMP (Adaptive Metadata-aware PhoBERT). "
        "Mô hình được thiết kế để kết hợp hiệu quả giữa khả năng trích xuất ngữ nghĩa văn bản của mô hình tiền huấn luyện PhoBERT và Multi-Kernel CNN "
        "với thông tin ngữ cảnh số (Metadata) đi kèm như điểm đánh giá (rating), số lượt thích (num_likes), và điểm chất lượng sản phẩm (quality)."
    )
    add_body_paragraph(
        "Dự án tập trung giải quyết các mục tiêu cụ thể sau:"
    )
    add_list_item("-", "Xây dựng pipeline tiền xử lý và tăng cường dữ liệu khép kín gồm 17 bước xử lý tự động dành riêng cho văn bản tiếng Việt.")
    add_list_item("-", "Đề xuất kiến trúc PhoBERT-AMP tích hợp cơ chế Cổng thích ứng Metadata (Adaptive Metadata Gate) giúp mô hình linh hoạt quyết định mức độ sử dụng thông tin bổ trợ theo từng mẫu bình luận.")
    add_list_item("-", "Tối ưu hóa quy trình huấn luyện cho môi trường phần cứng giới hạn VRAM (GPU 8GB) với các kỹ thuật Mixed Precision (AMP), Gradient Accumulation và Dynamic Padding.")
    add_list_item("-", "Đóng gói mô hình thành dịch vụ RESTful Microservice API với FastAPI và xây dựng phương án tích hợp thực tế với hệ thống Backend ASP.NET Core.")

    add_section_heading("1.3. Ý nghĩa thực tiễn và ứng dụng doanh nghiệp")
    add_body_paragraph(
        "Hệ thống sau khi hoàn thành có khả năng phân loại bình luận tự động thành 3 lớp cảm xúc: Tiêu cực (Negative), Bình thường (Normal), "
        "và Tích cực (Positive). Việc tích hợp mô hình vào hệ thống quản trị thương mại điện tử giúp tự động phân loại hàng ngàn đánh giá mỗi phút, "
        "lọc ra các phản hồi tiêu cực để chuyển tiếp ngay lập tức cho bộ phận chăm sóc khách hàng, góp phần nâng cao trải nghiệm người dùng."
    )

    # ---------------------------------------------------------
    # CHƯƠNG 2: KIẾN TRÚC MÔ HÌNH ĐỀ XẤU PHOBERT-AMP
    # ---------------------------------------------------------
    add_chapter_heading("CHƯƠNG 2: KIẾN TRÚC MÔ HÌNH ĐỀ XẤU PHOBERT-AMP")

    add_section_heading("2.1. Đặt vấn đề và hạn chế của các mô hình truyền thống")
    add_body_paragraph(
        "Các tiếp cận phân loại cảm xúc truyền thống thường sử dụng mô hình Bag-of-Words, TF-IDF kết hợp với SVM/Random Forest hoặc các mạng chuỗi "
        "như LSTM, GRU. Khi các mô hình Transformer như PhoBERT ra đời, hiệu năng xử lý văn bản đã tăng lên vượt bậc. Tuy nhiên, nếu chỉ sử dụng "
        "PhoBERT thuần túy hoặc nối cứng (Concatenate) thông tin Metadata với vector văn bản, mô hình sẽ gặp hai rủi ro lớn: (1) Bị phụ thuộc quá đà "
        "vào điểm rating (khiến mô hình bỏ qua văn bản), hoặc (2) Không học được biểu diễn riêng biệt cho từng trường số bổ trợ."
    )

    add_section_heading("2.2. Chi tiết kiến trúc mô hình PhoBERT-AMP đề xuất")
    add_body_paragraph(
        "Mô hình PhoBERT-AMP (Adaptive Metadata-aware PhoBERT) được thiết kế dạng song song đa nhánh nhằm kết hợp tối ưu giữa ngữ nghĩa văn bản "
        "và thông tin Metadata. Cấu trúc tổng thể của mô hình bao gồm 4 khối chức năng chính:"
    )

    add_subsection_heading("2.2.1. Nhánh xử lý văn bản (Text Branch)")
    add_list_item("1.", "PhoBERT Encoder: Chuỗi từ sau khi tách từ tiếng Việt được đưa qua PhoBERT (bản vinai/phobert-base) để trích xuất không gian biểu diễn ngữ cảnh (last_hidden_state).")
    add_list_item("2.", "Multi-Kernel CNN: Trọng số output của PhoBERT được đưa qua mạng CNN song song với 4 kích thước bộ lọc k ∈ [2, 3, 4, 5], mỗi loại 128 filters. Khối này đảm nhận trích xuất các cụm n-gram từ 2 đến 5 từ mang sắc thái cảm xúc mạnh.")
    add_list_item("3.", "Batch Normalization & Pooling: Ngay sau mỗi lớp Conv1D, lớp Batch Normalization (BatchNorm1d) được thêm vào để ổn định dải gradient, kết hợp hàm kích hoạt ReLU và Masked Global Max Pooling để chọn lọc đặc trưng nổi bật nhất, cho ra vector văn bản 512 chiều.")

    add_subsection_heading("2.2.2. Nhánh biểu diễn Metadata (Per-Field Embedding Branch)")
    add_body_paragraph(
        "Khác với việc biến đổi chung các trường số qua một lớp Dense duy nhất, PhoBERT-AMP áp dụng cơ chế Per-Field Embedding. Mỗi trường số "
        "(Rating, Số lượt thích, Điểm chất lượng) được đi qua một tuyến biến đổi độc lập bao gồm: Lớp tuyến tính Linear(1 -> 32) + ReLU + Dropout(0.2). "
        "Ba trường metadata tạo thành vector đặc trưng 96 chiều (3 x 32 = 96d), giúp mô hình duy trì biểu diễn riêng biệt cho từng loại thông tin trước khi hợp nhất."
    )

    add_subsection_heading("2.2.3. Cơ chế Cổng thích ứng Metadata (Adaptive Metadata Gate)")
    add_body_paragraph(
        "Cơ chế Gate đóng vai trò như một bộ điều tiết tự động, tính toán mức độ tin tưởng của Metadata đối với từng mẫu bình luận cụ thể dựa trên sự kết hợp "
        "giữa vector văn bản và vector metadata:"
    )
    
    # Mathematical formulation
    p_math = doc.add_paragraph()
    p_math.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_m = p_math.add_run(
        "gate_input = Concatenate(text_vector, metadata_vector)  [Chiều: 608]\n"
        "gate_score = min( Sigmoid( W2 · ReLU( W1 · gate_input + b1 ) + b2 ), 0.5 )\n"
        "gated_metadata = gate_score × metadata_vector"
    )
    r_m.font.name = 'Cambria Math'
    r_m.font.size = Pt(11)
    r_m.font.bold = True

    add_body_paragraph(
        "Cơ chế này đảm bảo: Đối với các câu văn có sắc thái rõ ràng ('sản phẩm chất lượng quá kém'), gate_score tự động giảm về tiệm cận 0 để mô hình "
        "dựa hoàn toàn vào văn bản. Ngược lại, với các câu mơ hồ ('dùng tạm được'), gate_score tăng lên mức tối đa (0.5) để bổ sung thông tin từ Metadata. "
        "Để tránh việc mô hình lạm dụng Metadata, hàm mất mát được bổ sung thành phần phạt L1 Regularization: L_penalty = λ · |gate_score| (với λ = 0.015)."
    )

    add_subsection_heading("2.2.4. Khối kết hợp và lớp phân loại (Fusion & Classification Head)")
    add_body_paragraph(
        "Vector văn bản (512d) và vector gated metadata (96d) được nối lại thành vector 608 chiều. Vector này đi qua lớp chuẩn hóa LayerNorm, "
        "tiếp tục qua lớp Fully Connected FC(256), hàm kích hoạt ReLU, Dropout(0.3) và lớp tuyến tính cuối cùng để sinh ra 3 giá trị Logits cho 3 nhãn cảm xúc."
    )

    add_section_heading("2.3. So sánh đối sánh mô hình cũ và PhoBERT-AMP")

    table_cmp = doc.add_table(rows=6, cols=3)
    table_cmp.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Thành phần / Kỹ thuật", "Mô hình cũ (Baseline)", "Mô hình PhoBERT-AMP (Đề xuất)"]
    
    for i, h in enumerate(headers):
        cell = table_cmp.cell(0, i)
        set_cell_background(cell, HEX_TABLE_HEADER_BG)
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.size = Pt(12)

    cmp_data = [
        ("Kiến trúc mạng CNN", "Conv1D -> ReLU -> MaxPool", "Conv1D -> BatchNorm -> ReLU -> Masked MaxPool"),
        ("Xử lý thuộc tính Metadata", "Lớp Dense chung (3 -> 64)", "Per-Field Embedding riêng biệt (1 -> 32 x 3 = 96d)"),
        ("Cơ chế kết hợp (Fusion)", "Nối cứng (Concatenate)", "Cổng thích ứng (Adaptive Metadata Gate ∈ 0~0.5)"),
        ("Chuẩn hóa đầu vào FC", "Không sử dụng", "Lớp LayerNorm trước FC Head"),
        ("Tổng chiều vector fusion", "576 (512 + 64)", "608 (512 + 96) kết hợp L1 Regularization")
    ]

    for row_idx, data in enumerate(cmp_data, start=1):
        for col_idx, text in enumerate(data):
            cell = table_cmp.cell(row_idx, col_idx)
            if row_idx % 2 == 1:
                set_cell_background(cell, HEX_TABLE_ALT_BG)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(11.5)
            if col_idx == 0:
                r.font.bold = True

    set_table_borders(table_cmp, color=HEX_BORDER, sz="6")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ---------------------------------------------------------
    # CHƯƠNG 3: QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU VÀ AUGMENTATION
    # ---------------------------------------------------------
    add_chapter_heading("CHƯƠNG 3: QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU VÀ AUGMENTATION")

    add_section_heading("3.1. Quy trình 17 bước tiền xử lý dữ liệu tự động")
    add_body_paragraph(
        "Để đảm bảo chất lượng dữ liệu đầu vào sạch và nhất quán, dự án đã xây dựng pipeline tiền xử lý gồm 17 bước liên hoàn trong module `scripts/prepare_data.py`:"
    )

    table_p17 = doc.add_table(rows=18, cols=3)
    table_p17.alignment = WD_TABLE_ALIGNMENT.CENTER
    p17_headers = ["STT", "Bước xử lý", "Mục tiêu và phương pháp thực hiện"]
    
    for i, h in enumerate(p17_headers):
        cell = table_p17.cell(0, i)
        set_cell_background(cell, HEX_TABLE_HEADER_BG)
        set_cell_margins(cell, top=100, bottom=100, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.size = Pt(12)

    steps_data = [
        ("1", "Loại bỏ trùng lặp", "Xoá các bản ghi trùng lặp hoàn toàn về nội dung văn bản."),
        ("2", "Loại bỏ dữ liệu rỗng", "Lọc bỏ các bình luận không chứa ký tự văn bản."),
        ("3", "Chuẩn hóa Unicode NFC", "Chuyển toàn bộ mã hóa tiếng Việt về chuẩn NFC thống nhất."),
        ("4", "Chuẩn hóa khoảng trắng", "Xóa ký tự xuống dòng, tab, đưa về khoảng trắng đơn."),
        ("5", "Chuẩn hóa chữ thường", "Chuyển toàn bộ chuỗi văn bản về định dạng viết thường."),
        ("6", "Chuẩn hóa Teencode", "Áp dụng từ điển teencode: 'ko/k' -> 'không', 'm' -> 'mình'..."),
        ("7", "Chuẩn hóa dấu câu", "Thu gọn chuỗi dấu câu lặp như '????' -> '?', '!!!!' -> '!'."),
        ("8", "Chuẩn hóa từ viết tắt", "Áp dụng từ điển TMĐT: 'sp' -> 'sản phẩm', 'dmel' -> 'đẹp lắm'..."),
        ("9", "Chuẩn hóa Emoji", "Chuyển emoji thành token cảm xúc: 😊 -> 'tích cực', 😡 -> 'tiêu cực'."),
        ("10", "Chuẩn hóa Icon text", "Chuyển icon gõ phím ':))', '^^', 'TT' thành token cảm xúc tương ứng."),
        ("11", "Chuẩn hóa ký tự lặp", "Thu gọn ký tự lặp như 'đẹpuuu' -> 'đẹpu' (giữ lặp 2 lần cho từ tiếng Anh)."),
        ("12", "Thay thế URL link", "Thay thế các đường dẫn web thành token đại diện <URL>."),
        ("13", "Loại bỏ thẻ HTML", "Bóc tách và xoá bỏ các thẻ HTML rác còn sót lại."),
        ("14", "Word Segmentation", "Tách từ ghép tiếng Việt bằng công cụ pyvi (vd: 'sản phẩm' -> 'sản_phẩm')."),
        ("15", "PhoBERT Tokenization", "Mã hóa câu thành chuỗi token ID và attention mask."),
        ("16", "Chuẩn hóa Metadata", "Biến đổi Log-transform num_likes, StandardScale rating và quality."),
        ("17", "Data Augmentation", "Áp dụng kỹ thuật EDA cho tập huấn luyện để cân bằng lớp.")
    ]

    for row_idx, (stt, name, desc) in enumerate(steps_data, start=1):
        c0 = table_p17.cell(row_idx, 0)
        c1 = table_p17.cell(row_idx, 1)
        c2 = table_p17.cell(row_idx, 2)
        c0.width = Inches(0.6)
        c1.width = Inches(2.2)
        c2.width = Inches(3.7)
        
        if row_idx % 2 == 1:
            set_cell_background(c0, HEX_TABLE_ALT_BG)
            set_cell_background(c1, HEX_TABLE_ALT_BG)
            set_cell_background(c2, HEX_TABLE_ALT_BG)
        set_cell_margins(c0, top=60, bottom=60, left=80, right=80)
        set_cell_margins(c1, top=60, bottom=60, left=80, right=80)
        set_cell_margins(c2, top=60, bottom=60, left=80, right=80)
        
        p0 = c0.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(stt)
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(11)
        
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(name)
        r1.font.name = 'Times New Roman'
        r1.font.bold = True
        r1.font.size = Pt(11)
        
        p2 = c2.paragraphs[0]
        p2.paragraph_format.space_after = Pt(0)
        r2 = p2.add_run(desc)
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(11)

    set_table_borders(table_p17, color=HEX_BORDER, sz="6")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    add_section_heading("3.2. Kỹ thuật tăng cường dữ liệu EDA cho lớp thiểu số")
    add_body_paragraph(
        "Nhằm giải quyết vấn đề mất cân bằng giữa các lớp cảm xúc mà không gây overfit, kỹ thuật Easy Data Augmentation (EDA) "
        "chỉ được áp dụng riêng cho tập Train sau khi đã phân chia tập dữ liệu. Các thao tác bao gồm: Thay thế từ đồng nghĩa (SR), "
        "Chèn ngẫu nhiên từ đồng nghĩa (RI), Hoán đổi vị trí từ (RS), và Xóa ngẫu nhiên từ (RD) với xác suất p = 0.1."
    )

    # ---------------------------------------------------------
    # CHƯƠNG 4: THỐNG KÊ TẬP DỮ LIỆU VÀ THIẾT LẬP THỰC NGHIỆM
    # ---------------------------------------------------------
    add_chapter_heading("CHƯƠNG 4: THỐNG KÊ TẬP DỮ LIỆU VÀ THIẾT LẬP THỰC NGHIỆM")

    add_section_heading("4.1. Phân bố tập dữ liệu gốc")
    add_body_paragraph(
        "Tập dữ liệu thu thập gồm 15.378 bình luận tiếng Việt được thu thập từ các sàn thương mại điện tử lớn. Phân bố cụ thể như sau:"
    )

    table_ds = doc.add_table(rows=4, cols=4)
    table_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    ds_headers = ["Nhãn cảm xúc", "Số lượng mẫu", "Tỷ lệ phần trăm", "Đặc điểm nhận diện"]
    
    for i, h in enumerate(ds_headers):
        cell = table_ds.cell(0, i)
        set_cell_background(cell, HEX_TABLE_HEADER_BG)
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.size = Pt(12)

    ds_rows = [
        ("Positive (Tích cực)", "6.305", "41,0%", "Khen ngợi chất lượng, giao hàng nhanh, giới thiệu mua."),
        ("Normal (Bình thường)", "5.175", "33,7%", "Nội dung trung tính, xác nhận nhận hàng, bình luận ngắn."),
        ("Negative (Tiêu cực)", "3.898", "25,3%", "Chê chất lượng, hàng hỏng, thái độ phục vụ kém.")
    ]

    for row_idx, (l, c, p_val, d) in enumerate(ds_rows, start=1):
        table_ds.cell(row_idx, 0).paragraphs[0].add_run(l).font.bold = True
        table_ds.cell(row_idx, 1).paragraphs[0].add_run(c)
        table_ds.cell(row_idx, 2).paragraphs[0].add_run(p_val)
        table_ds.cell(row_idx, 3).paragraphs[0].add_run(d)
        
        for col_idx in range(4):
            c_cell = table_ds.cell(row_idx, col_idx)
            if row_idx % 2 == 1:
                set_cell_background(c_cell, HEX_TABLE_ALT_BG)
            set_cell_margins(c_cell, top=80, bottom=80, left=100, right=100)
            p_c = c_cell.paragraphs[0]
            p_c.runs[0].font.name = 'Times New Roman'
            p_c.runs[0].font.size = Pt(11.5)
            if col_idx in [1, 2]:
                p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER

    set_table_borders(table_ds, color=HEX_BORDER, sz="6")
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    report_dir = r"c:\DevLTDD\Đồ án tốt nghiệp\model_phobert_metadata_cnn\phobert_cnn_metadata_sentiment\outputs\reports"
    add_image_with_caption(os.path.join(report_dir, "dataset_distribution.png"), "Biểu đồ phân bố các lớp cảm xúc trước và sau khi áp dụng Augmentation", figure_num="4.1", width_inches=5.8)

    add_section_heading("4.2. Phân chia tập dữ liệu và kết quả cân bằng lớp")
    add_body_paragraph(
        "Tập dữ liệu được chia theo phương pháp Stratified Split thành 3 tập riêng biệt: Tập Train (70%), Tập Validation (15%), và Tập Test (15%). "
        "Sau khi áp dụng Data Augmentation cho tập Train, số lượng mẫu giữa 3 lớp đạt trạng thái cân bằng hoàn toàn với 4.413 mẫu cho mỗi lớp "
        "(tổng cộng 13.239 mẫu tập Train). Tập Validation giữ nguyên 2.306 mẫu và Tập Test giữ nguyên 2.308 mẫu để đảm bảo tính khách quan."
    )

    add_section_heading("4.3. Cấu hình thông số huấn luyện")
    add_body_paragraph(
        "Quá trình huấn luyện được cấu hình tối ưu để chạy mượt mà trên phần cứng GPU 8GB VRAM:"
    )

    table_cfg = doc.add_table(rows=8, cols=3)
    table_cfg.alignment = WD_TABLE_ALIGNMENT.CENTER
    cfg_headers = ["Thông số huấn luyện", "Giá trị thiết lập", "Ý nghĩa và vai trò kỹ thuật"]
    
    for i, h in enumerate(cfg_headers):
        cell = table_cfg.cell(0, i)
        set_cell_background(cell, HEX_TABLE_HEADER_BG)
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.size = Pt(12)

    cfg_data = [
        ("PhoBERT Learning Rate", "2e-5", "Fine-tune nhẹ nhàng các lớp pre-trained của PhoBERT."),
        ("Head Learning Rate", "1e-3", "Cho phép các lớp CNN, Per-Field Embedding và FC học nhanh hơn."),
        ("Batch Size & Accumulation", "8 x 4 (Effective = 32)", "Mô phỏng batch size 32 thực tế mà không gây quá tải VRAM."),
        ("Mixed Precision (AMP)", "Enabled (FP16)", "Tăng gấp đôi tốc độ huấn luyện và giảm 50% lượng VRAM tiêu thụ."),
        ("Sequence Max Length", "256 tokens", "Đảm bảo bao phủ 99,5% độ dài bình luận thực tế."),
        ("Optimizer & Scheduler", "AdamW + Warmup Linear", "Warmup 10% tổng số bước để ổn định gradient ban đầu."),
        ("Early Stopping Patience", "4 Epochs (Macro F1)", "Tự động dừng khi chỉ số F1 tập validation không cải thiện.")
    ]

    for row_idx, (p_name, val, reason) in enumerate(cfg_data, start=1):
        c0 = table_cfg.cell(row_idx, 0)
        c1 = table_cfg.cell(row_idx, 1)
        c2 = table_cfg.cell(row_idx, 2)
        
        if row_idx % 2 == 1:
            set_cell_background(c0, HEX_TABLE_ALT_BG)
            set_cell_background(c1, HEX_TABLE_ALT_BG)
            set_cell_background(c2, HEX_TABLE_ALT_BG)
        set_cell_margins(c0, top=60, bottom=60, left=80, right=80)
        set_cell_margins(c1, top=60, bottom=60, left=80, right=80)
        set_cell_margins(c2, top=60, bottom=60, left=80, right=80)
        
        r0 = c0.paragraphs[0].add_run(p_name)
        r0.font.bold = True
        r1 = c1.paragraphs[0].add_run(val)
        c1.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = c2.paragraphs[0].add_run(reason)
        
        for r_item in [r0, r1, r2]:
            r_item.font.name = 'Times New Roman'
            r_item.font.size = Pt(11.5)

    set_table_borders(table_cfg, color=HEX_BORDER, sz="6")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ---------------------------------------------------------
    # CHƯƠNG 5: KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ
    # ---------------------------------------------------------
    add_chapter_heading("CHƯƠNG 5: KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ MÔ HÌNH")

    add_section_heading("5.1. Kết quả tổng quát trên tập kiểm thử (Test Set)")
    add_body_paragraph(
        "Mô hình PhoBERT-AMP sau khi hoàn thành huấn luyện đã được đánh giá trên tập kiểm thử độc lập gồm 2.308 mẫu bình luận. "
        "Kết quả đánh giá tổng quát được thể hiện ở bảng dưới đây:"
    )

    table_res = doc.add_table(rows=2, cols=4)
    table_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_headers = ["Accuracy", "Macro F1-Score", "Macro Precision", "Macro Recall"]
    res_values = ["99,61%", "99,56%", "99,56%", "99,57%"]
    
    for i in range(4):
        c_hdr = table_res.cell(0, i)
        c_val = table_res.cell(1, i)
        set_cell_background(c_hdr, HEX_TABLE_HEADER_BG)
        set_cell_background(c_val, HEX_TABLE_ALT_BG)
        set_cell_margins(c_hdr, top=80, bottom=80, left=60, right=60)
        set_cell_margins(c_val, top=100, bottom=100, left=60, right=60)
        
        p0 = c_hdr.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r0 = p0.add_run(res_headers[i])
        r0.font.name = 'Times New Roman'
        r0.font.bold = True
        r0.font.size = Pt(12)
        
        p1 = c_val.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p1.add_run(res_values[i])
        r1.font.name = 'Times New Roman'
        r1.font.bold = True
        r1.font.size = Pt(13)

    set_table_borders(table_res, color=HEX_BORDER, sz="6")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    add_section_heading("5.2. Kết quả đánh giá chi tiết theo từng lớp cảm xúc")

    table_cls = doc.add_table(rows=4, cols=5)
    table_cls.alignment = WD_TABLE_ALIGNMENT.CENTER
    cls_headers = ["Nhãn cảm xúc", "Precision", "Recall", "F1-Score", "Số lượng mẫu (Support)"]
    
    for i, h in enumerate(cls_headers):
        cell = table_cls.cell(0, i)
        set_cell_background(cell, HEX_TABLE_HEADER_BG)
        set_cell_margins(cell, top=100, bottom=100, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.size = Pt(12)

    cls_rows = [
        ("Negative (Tiêu cực)", "99,32%", "99,32%", "99,32%", "585"),
        ("Normal (Bình thường)", "99,36%", "99,49%", "99,42%", "777"),
        ("Positive (Tích cực)", "100,00%", "99,89%", "99,95%", "946")
    ]

    for row_idx, row_data in enumerate(cls_rows, start=1):
        for col_idx, text_val in enumerate(row_data):
            cell = table_cls.cell(row_idx, col_idx)
            if row_idx % 2 == 1:
                set_cell_background(cell, HEX_TABLE_ALT_BG)
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text_val)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(11.5)
            if col_idx == 0:
                r.font.bold = True
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    set_table_borders(table_cls, color=HEX_BORDER, sz="6")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    add_image_with_caption(os.path.join(report_dir, "confusion_matrix.png"), "Ma trận nhầm lẫn (Confusion Matrix) đánh giá trên tập kiểm thử Test Set", figure_num="5.1", width_inches=5.2)
    add_image_with_caption(os.path.join(report_dir, "enhanced_training_curves.png"), "Đường cong huấn luyện thể hiện sự biến thiên của Loss, Accuracy, F1 và Gate Score qua các epoch", figure_num="5.2", width_inches=5.8)
    add_image_with_caption(os.path.join(report_dir, "per_class_metrics.png"), "Biểu đồ so sánh các chỉ số Precision, Recall và F1-Score giữa các nhãn cảm xúc", figure_num="5.3", width_inches=5.2)

    add_section_heading("5.3. Phân tích thực tế hoạt động của Gate Score")
    add_body_paragraph(
        "Qua theo dõi quá trình huấn luyện và suy luận, giá trị trung bình của Gate Score dao động trong khoảng từ 0.35 đến 0.42. "
        "Điều này chứng tỏ mô hình đã học được cách kết hợp hài hòa: Dựa vào văn bản làm tín hiệu phân loại chính (chiếm khoảng 65% trọng số), "
        "đồng thời sử dụng thông tin Metadata để hỗ trợ giải quyết các trường hợp bình luận ngắn hoặc có sắc thái không rõ ràng."
    )

    # ---------------------------------------------------------
    # CHƯƠNG 6: TRIỂN KHAI MICROSERVICE API VÀ TÍCH HỢP
    # ---------------------------------------------------------
    add_chapter_heading("CHƯƠNG 6: TRIỂN KHAI MICROSERVICE API VÀ TÍCH HỢP HỆ THỐNG")

    add_section_heading("6.1. Thiết kế kiến trúc Microservice API")
    add_body_paragraph(
        "Mô hình PhoBERT-AMP được đóng gói thành một dịch vụ Microservice độc lập sử dụng framework FastAPI. Dịch vụ cung cấp các RESTful API "
        "cho phép các hệ thống ứng dụng bên ngoài dễ dàng gửi dữ liệu bình luận và nhận lại kết quả phân loại cảm xúc theo thời gian thực."
    )

    table_api = doc.add_table(rows=4, cols=4)
    table_api.alignment = WD_TABLE_ALIGNMENT.CENTER
    api_headers = ["Endpoint URL", "Phương thức", "Mức độ ưu tiên", "Mô tả chức năng chính"]
    
    for i, h in enumerate(api_headers):
        cell = table_api.cell(0, i)
        set_cell_background(cell, HEX_TABLE_HEADER_BG)
        set_cell_margins(cell, top=100, bottom=100, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = 'Times New Roman'
        r.font.bold = True
        r.font.size = Pt(12)

    api_rows = [
        ("/predict", "POST", "Bắt buộc", "Dự đoán cảm xúc cho một bình luận đơn lẻ kèm metadata."),
        ("/predict/batch", "POST", "Khuyên dùng", "Dự đoán cảm xúc hàng loạt cho một danh sách các bình luận."),
        ("/health", "GET", "Hệ thống", "Kiểm tra trạng thái sẵn sàng của dịch vụ và mô hình.")
    ]

    for row_idx, (ep, mth, prio, desc) in enumerate(api_rows, start=1):
        table_api.cell(row_idx, 0).paragraphs[0].add_run(ep).font.bold = True
        table_api.cell(row_idx, 1).paragraphs[0].add_run(mth)
        table_api.cell(row_idx, 2).paragraphs[0].add_run(prio)
        table_api.cell(row_idx, 3).paragraphs[0].add_run(desc)
        
        for col_idx in range(4):
            c_cell = table_api.cell(row_idx, col_idx)
            if row_idx % 2 == 1:
                set_cell_background(c_cell, HEX_TABLE_ALT_BG)
            set_cell_margins(c_cell, top=80, bottom=80, left=80, right=80)
            p_c = c_cell.paragraphs[0]
            p_c.runs[0].font.name = 'Times New Roman'
            p_c.runs[0].font.size = Pt(11.5)
            if col_idx in [1, 2]:
                p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER

    set_table_borders(table_api, color=HEX_BORDER, sz="6")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    add_section_heading("6.2. Mã nguồn mẫu tích hợp từ C# (ASP.NET Core)")
    add_body_paragraph(
        "Đoạn mã C# dưới đây minh họa phương thức kết nối và gửi yêu cầu dự đoán cảm xúc từ ứng dụng Web Backend tới Microservice API:"
    )

    p_cs = doc.add_paragraph()
    p_cs.paragraph_format.left_indent = Inches(0.4)
    p_cs.paragraph_format.right_indent = Inches(0.4)
    p_cs.paragraph_format.space_after = Pt(8)
    
    r_cs = p_cs.add_run(
        "public class SentimentClient\n"
        "{\n"
        "    private readonly HttpClient _http;\n"
        "    public SentimentClient(HttpClient http) => _http = http; // Base URL: http://localhost:8000\n\n"
        "    public async Task<SentimentResult> PredictAsync(string text, double? rating = null, int? likes = null)\n"
        "    {\n"
        "        var payload = new { text, rating, num_likes = likes };\n"
        "        var response = await _http.PostAsJsonAsync(\"/predict\", payload);\n"
        "        response.EnsureSuccessStatusCode();\n"
        "        return await response.Content.ReadFromJsonAsync<SentimentResult>();\n"
        "    }\n"
        "}\n\n"
        "public record SentimentResult(string Text, string Label, double Confidence,\n"
        "                               Dictionary<string, double> Probabilities, double? GateScore);"
    )
    r_cs.font.name = 'Consolas'
    r_cs.font.size = Pt(9.5)

    # ---------------------------------------------------------
    # CHƯƠNG 7: CẤU TRÚC MÃ NGUỒN VÀ HƯỚNG DẪN VẬN HÀNH
    # ---------------------------------------------------------
    add_chapter_heading("CHƯƠNG 7: CẤU TRÚC MÃ NGUỒN VÀ HƯỚNG DẪN VẬN HÀNH")

    add_section_heading("7.1. Thống kê cấu trúc cây thư mục dự án")
    p_tree = doc.add_paragraph()
    p_tree.paragraph_format.left_indent = Inches(0.4)
    r_tree = p_tree.add_run(
        "phobert_cnn_metadata_sentiment/\n"
        "├── config.py                    # File cấu hình thông số tập trung toàn hệ thống\n"
        "├── requirements.txt             # Danh sách thư viện phụ thuộc của dự án\n"
        "├── setup.py                     # Cấu hình cài đặt package nội bộ\n"
        "├── data/\n"
        "│   ├── raw/comments.csv         # File dữ liệu thô 15.378 dòng\n"
        "│   └── processed/               # Dữ liệu sau tiền xử lý (train/val/test.csv)\n"
        "├── src/\n"
        "│   ├── preprocessing/           # Module làm sạch văn bản, từ điển teencode, tách từ\n"
        "│   ├── augmentation/            # Module tăng cường dữ liệu EDA\n"
        "│   ├── dataset.py               # Lớp Dataset PyTorch và Collate Function\n"
        "│   ├── model.py                 # Định nghĩa kiến trúc PhoBERT-AMP\n"
        "│   ├── train.py                 # Script điều khiển quá trình huấn luyện\n"
        "│   ├── evaluate.py              # Script đánh giá mô hình trên tập Test\n"
        "│   └── predict.py               # Module hỗ trợ suy luận dự đoán thực tế\n"
        "├── api/\n"
        "│   ├── app.py                   # REST API Service phát triển bằng FastAPI\n"
        "│   └── index.html               # Giao diện kiểm thử trực quan\n"
        "└── outputs/\n"
        "    ├── checkpoints/             # Nơi lưu trữ checkpoint tốt nhất (best_model.pt)\n"
        "    └── reports/                 # Nơi lưu trữ biểu đồ và báo cáo kết quả"
    )
    r_tree.font.name = 'Consolas'
    r_tree.font.size = Pt(9)

    add_section_heading("7.2. Các bước vận hành hệ thống")
    add_list_item("Bước 1:", "Chuyển đổi dữ liệu gốc từ Excel sang CSV: python -m scripts.convert_excel_to_csv")
    add_list_item("Bước 2:", "Thực thi pipeline tiền xử lý 17 bước: python -m scripts.prepare_data")
    add_list_item("Bước 3:", "Huấn luyện mô hình PhoBERT-AMP: python -m src.train")
    add_list_item("Bước 4:", "Đánh giá hiệu năng trên tập Test: python -m src.evaluate")
    add_list_item("Bước 5:", "Khởi chạy Microservice API: uvicorn api.app:app --port 8000 --reload")

    # ---------------------------------------------------------
    # CHƯƠNG 8: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
    # ---------------------------------------------------------
    add_chapter_heading("CHƯƠNG 8: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")

    add_section_heading("8.1. Kết luận")
    add_body_paragraph(
        "Đề tài đã nghiên cứu và phát triển thành công mô hình học sâu PhoBERT-AMP áp dụng cho bài toán phân loại cảm xúc bình luận tiếng Việt "
        "trong thương mại điện tử. Việc tích hợp cơ chế Cổng thích ứng Metadata (Adaptive Metadata Gate) đã chứng minh tính đúng đắn và hiệu quả vượt trội, "
        "đạt độ chính xác 99,61% và F1-Score 99,56%. Hệ thống được tối ưu hóa cho phần cứng thực tế và sẵn sàng triển khai dưới dạng dịch vụ REST API."
    )

    add_section_heading("8.2. Hướng phát triển tiếp theo")
    add_list_item("a)", "Mở rộng thu thập và tích hợp thêm các trường dữ liệu bổ trợ như thời gian đăng bình luận, số lượt phản hồi, lịch sử mua hàng.")
    add_list_item("b)", "Thử nghiệm nâng cấp mô hình nền lên PhoBERT Large hoặc các mô hình ngôn ngữ lớn (LLMs) tiếng Việt.")
    add_list_item("c)", "Đóng gói toàn bộ mô hình và dịch vụ API dưới dạng Docker Container để hỗ trợ tự động mở rộng quy mô (Auto-scaling) trên hạ tầng Cloud.")

    doc.add_paragraph().paragraph_format.space_after = Pt(24)
    p_end = doc.add_paragraph()
    p_end.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_end = p_end.add_run("--- HẾT BÁO CÁO ---")
    r_end.font.name = 'Times New Roman'
    r_end.font.bold = True

    # Save documents
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(repo_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    out_file = os.path.join(docs_dir, "Báo_cáo_chi_tiết_Dự_án_PhoBERT_AMP.docx")
    doc.save(out_file)
    print(f"Academic style report generated successfully at: {out_file}")

if __name__ == "__main__":
    create_report()
