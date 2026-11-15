#!/usr/bin/env python3
import shutil
from pathlib import Path

PROJECT_ROOT = Path('/mnt/wav2vec2/khoan/source_code/RB-Ortho/ChronosArt')
out_dir = PROJECT_ROOT / 'output/fairness_study'
out_dir.mkdir(parents=True, exist_ok=True)

# Copy files
shutil.copy2(PROJECT_ROOT / 'styles/08_pencil_sketch.png', out_dir / 'style_ref.png')
shutil.copy2(PROJECT_ROOT / 'output/vanilla_test/vanilla_pure_kettle.png', out_dir / 'vanilla_pure.png')
shutil.copy2(PROJECT_ROOT / 'output/vanilla_test/vanilla_styled_kettle.png', out_dir / 'vanilla_styled.png')
shutil.copy2(PROJECT_ROOT / 'output/experiment_pure_prompt/kettle_baseline_pure.png', out_dir / 'baseline_pure.png')
shutil.copy2(PROJECT_ROOT / 'output/experiment_pure_prompt/kettle_chronos_pure.png', out_dir / 'chronos_pure.png')

html = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phân Tích Tính Khách Quan: Prompt Có Style Text vs Pure Image-Driven Style Transfer</title>
    <style>
        :root {
            --bg: #090d14;
            --card-bg: #121824;
            --border: #222d3d;
            --text-main: #f0f6fc;
            --text-sub: #8b949e;
            --accent: #58a6ff;
            --warn: #e3b341;
            --success: #3fb950;
            --danger: #f85149;
            --purple: #a371f7;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 30px;
        }
        .header {
            max-width: 1500px;
            margin: 0 auto 24px auto;
        }
        .header h1 {
            font-size: 26px;
            background: linear-gradient(90deg, #e3b341, #58a6ff, #3fb950);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .insight-box {
            background: rgba(227, 179, 65, 0.1);
            border: 1px solid rgba(227, 179, 65, 0.4);
            border-radius: 10px;
            padding: 18px 22px;
            margin-top: 14px;
            font-size: 14px;
            line-height: 1.6;
        }
        .container {
            max-width: 1500px;
            margin: 24px auto 0 auto;
            display: flex;
            flex-direction: column;
            gap: 28px;
        }
        .style-banner {
            display: flex;
            align-items: center;
            gap: 20px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
        }
        .style-banner img {
            width: 110px;
            height: 110px;
            border-radius: 8px;
            object-fit: cover;
            border: 2px solid var(--accent);
        }
        .grid-cards {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .card-header {
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 8px;
            text-align: center;
            width: 100%;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border);
        }
        .card-sub {
            font-size: 11px;
            color: var(--text-sub);
            margin-bottom: 10px;
            text-align: center;
            min-height: 32px;
        }
        .img-wrap {
            width: 100%;
            aspect-ratio: 1/1;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            cursor: pointer;
        }
        .img-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.2s;
        }
        .img-wrap:hover img { transform: scale(1.03); }
        .score-list {
            margin-top: 12px;
            width: 100%;
            font-size: 12px;
            background: rgba(0,0,0,0.3);
            border-radius: 6px;
            padding: 8px 10px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .score-row {
            display: flex;
            justify-content: space-between;
        }
        .modal {
            display: none; position: fixed; z-index: 999;
            left: 0; top: 0; width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.88); backdrop-filter: blur(5px);
            align-items: center; justify-content: center; flex-direction: column;
        }
        .modal img { max-width: 85%; max-height: 85%; border-radius: 8px; }
        .modal-caption { margin-top: 14px; font-size: 15px; color: var(--text-main); font-weight: 600; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Bản Chất Khoa Học: Prompt Có Style Text vs Chuyển Phong Cách Thuần Từ Ảnh (Image-Driven)</h1>
        
        <div class="insight-box">
            <strong style="color: var(--warn); font-size: 15px;">Chất vấn xác đáng của bạn:</strong><br>
            <em>"Nếu câu prompt đã chứa 'in pencil sketch style' thì là full prompt rồi, bản thân Text-to-Image model tự động vẽ ra được chứ cần gì can thiệp nữa!"</em><br><br>
            <strong>Kết quả thực nghiệm khẳng định 100% nhận định này:</strong><br>
            &bull; Khi đưa prompt <code>"A kettle in pencil sketch style"</code> cho <strong>Base Model thuần (không dùng ảnh style, không PFB, không SAC)</strong>, điểm $S_{\text{harmonic}}$ đã đạt tới <strong style="color: var(--success);">0.4624</strong>! Bản thân Flan-T5 + Infinity đã tự vẽ tranh chì theo dữ liệu pre-training.<br>
            &bull; <strong>Ý nghĩa thực sự của Style Personalization:</strong> Không phải gõ chữ mô tả phong cách mà người dùng đưa vào <strong>một ảnh mẫu duy nhất</strong> (đặc biệt là tranh vẽ tay, phong cách dị biệt không có tên gọi trong từ điển) và prompt thuần <code>"A kettle"</code>. Mô hình phải trích xuất phong cách từ ảnh đó đắp lên ấm mà không được rò rỉ nội dung của ảnh mẫu!
        </div>
    </div>

    <div class="container">
        <div class="style-banner">
            <img src="style_ref.png" alt="Style Ref">
            <div>
                <h3 style="font-size: 16px; margin-bottom: 4px;">Ảnh Phong Cách Mẫu: 08_pencil_sketch.png</h3>
                <p style="font-size: 13px; color: var(--text-sub);">Thực chất là <strong>bức chân dung một phụ nữ vẽ bằng bút chì</strong>. Chứa cả <em>nét chì (style)</em> lẫn <em>khuôn mặt phụ nữ (content của style)</em>.</p>
            </div>
        </div>

        <div class="grid-cards">

            <!-- Card 1 -->
            <div class="card">
                <div class="card-header" style="color: var(--text-sub);">
                    1. Base Model Thuần (Prompt Gốc)
                </div>
                <div class="card-sub">
                    Prompt: <code>"A kettle"</code><br>
                    (Không ảnh style, không can thiệp)
                </div>
                <div class="img-wrap" onclick="openModal('vanilla_pure.png', 'Base Model Thuần: A kettle')">
                    <img src="vanilla_pure.png" alt="Vanilla Pure">
                </div>
                <div class="score-list">
                    <div class="score-row"><span>S_harmonic:</span> <b>0.3481</b></div>
                    <div class="score-row"><span>S_img (style chì):</span> <span>0.4965</span></div>
                    <div class="score-row"><span>S_txt (với "A kettle"):</span> <span>0.2990</span></div>
                    <div class="score-row" style="color: var(--danger); font-size: 11px;">Mô tả: Ấm kim loại/sứ thực tế có màu.</div>
                </div>
            </div>

            <!-- Card 2 -->
            <div class="card" style="border-color: rgba(227, 179, 65, 0.4);">
                <div class="card-header" style="color: var(--warn);">
                    2. Base Model (Text Prompt Đầy Đủ)
                </div>
                <div class="card-sub">
                    Prompt: <code>"A kettle in pencil sketch style"</code><br>
                    (Không dùng ảnh style, chỉ dùng text)
                </div>
                <div class="img-wrap" onclick="openModal('vanilla_styled.png', 'Base Model: A kettle in pencil sketch style')">
                    <img src="vanilla_styled.png" alt="Vanilla Styled">
                </div>
                <div class="score-list">
                    <div class="score-row"><span>S_harmonic:</span> <b style="color: var(--success);">0.4624</b></div>
                    <div class="score-row"><span>S_img (style chì):</span> <span>0.5842</span></div>
                    <div class="score-row"><span>S_txt (prompt full):</span> <span>0.3826</span></div>
                    <div class="score-row" style="color: var(--warn); font-size: 11px;">Mô tả: Model tự vẽ tranh chì theo trí nhớ pre-train!</div>
                </div>
            </div>

            <!-- Card 3 -->
            <div class="card" style="border-color: rgba(248, 81, 73, 0.4);">
                <div class="card-header" style="color: var(--danger);">
                    3. Baseline PFB + SAC (Pure Prompt)
                </div>
                <div class="card-sub">
                    Prompt: <code>"A kettle"</code> + Dùng Ảnh Style Mẫu<br>
                    (Paper baseline PFB ở s=3)
                </div>
                <div class="img-wrap" onclick="openModal('baseline_pure.png', 'Baseline PFB+SAC: A kettle')">
                    <img src="baseline_pure.png" alt="Baseline Pure">
                </div>
                <div class="score-list">
                    <div class="score-row"><span>S_harmonic:</span> <b>0.4224</b></div>
                    <div class="score-row"><span>S_img (style chì):</span> <span>0.6105</span></div>
                    <div class="score-row"><span>S_txt (với "A kettle"):</span> <span>0.3229</span></div>
                    <div class="score-row" style="color: var(--danger); font-size: 11px;">Mô tả: Bị rò rỉ nét mặt phụ nữ vào thân ấm!</div>
                </div>
            </div>

            <!-- Card 4 -->
            <div class="card" style="border-color: rgba(163, 113, 247, 0.6); background: rgba(163, 113, 247, 0.05);">
                <div class="card-header" style="color: var(--purple);">
                    4. ChronosArt (Pure Prompt)
                </div>
                <div class="card-sub">
                    Prompt: <code>"A kettle"</code> + Dùng Ảnh Style Mẫu<br>
                    (Triệt tiêu rò rỉ nội dung bằng Chiếu trực giao)
                </div>
                <div class="img-wrap" onclick="openModal('chronos_pure.png', 'ChronosArt: A kettle (Pure Prompt)')">
                    <img src="chronos_pure.png" alt="ChronosArt Pure">
                </div>
                <div class="score-list">
                    <div class="score-row"><span>S_harmonic:</span> <b>0.4076</b></div>
                    <div class="score-row"><span>S_img (style chì):</span> <span>0.5474</span></div>
                    <div class="score-row"><span>S_txt (với "A kettle"):</span> <span>0.3247</span></div>
                    <div class="score-row" style="color: var(--purple); font-size: 11px;">Mô tả: Ấm chì phác thảo sạch, không dính mặt người!</div>
                </div>
            </div>

        </div>
    </div>

    <div id="imageModal" class="modal" onclick="closeModal()">
        <img id="modalImg" src="">
        <div id="modalCaption" class="modal-caption"></div>
    </div>

    <script>
        function openModal(src, caption) {
            document.getElementById('modalImg').src = src;
            document.getElementById('modalCaption').innerText = caption;
            document.getElementById('imageModal').style.display = 'flex';
        }
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
    </script>
</body>
</html>
"""

(out_dir / 'index.html').write_text(html, encoding='utf-8')
print("Viewer built at: output/fairness_study/index.html")
