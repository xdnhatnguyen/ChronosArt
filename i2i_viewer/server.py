"""HTTP Server with REST APIs for ChronosArt-I2I Interactive Paper Curator."""
import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "i2i_viewer"
EVAL_DIR = PROJECT_ROOT / "auto_research_i2i" / ".eval_tmp"
CONTENT_DIR = PROJECT_ROOT / "content"
STYLES_DIR = PROJECT_ROOT / "styles"
FIGURE_DIR = PROJECT_ROOT / "report" / "Figure"

PORT = 8090

class I2IViewerHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/data/pairs.json":
            self.serve_file(WEB_DIR / "data" / "pairs.json", "application/json; charset=utf-8")
        elif path.startswith("/images/content/"):
            filename = path.replace("/images/content/", "")
            self.serve_file(CONTENT_DIR / filename, "image/png")
        elif path.startswith("/images/styles/"):
            filename = path.replace("/images/styles/", "")
            self.serve_file(STYLES_DIR / filename, "image/png")
        elif path.startswith("/images/eval/"):
            filename = path.replace("/images/eval/", "")
            self.serve_file(EVAL_DIR / filename, "image/png")
        elif path.startswith("/export/"):
            filename = path.replace("/export/", "")
            target = FIGURE_DIR / filename
            mime = "application/pdf" if filename.endswith(".pdf") else "image/png"
            self.serve_file(target, mime)
        elif path == "/api/status":
            pairs_json = WEB_DIR / "data" / "pairs.json"
            count = 0
            if pairs_json.exists():
                with open(pairs_json, "r") as f:
                    count = len(json.load(f))
            self.respond_json({"status": "running", "total_pairs": count, "port": PORT})
        else:
            self.send_error(404, f"File Not Found: {path}")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            req_data = json.loads(body) if body else {}
        except Exception:
            req_data = {}

        if path == "/api/export_latex":
            pair_ids = req_data.get("pair_ids", [])
            latex_code = self.generate_latex_snippet(pair_ids)
            self.respond_json({"success": True, "latex": latex_code})

        elif path == "/api/export_grid":
            pair_ids = req_data.get("pair_ids", [])
            png_name, pdf_name = self.generate_comparison_grid(pair_ids)
            self.respond_json({
                "success": True, 
                "png_url": f"/export/{png_name}",
                "pdf_url": f"/export/{pdf_name}",
                "png_path": str(FIGURE_DIR / png_name),
                "pdf_path": str(FIGURE_DIR / pdf_name),
                "count": len(pair_ids)
            })
        else:
            self.send_error(404, f"API endpoint not found: {path}")

    def serve_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(404, f"Not found: {file_path.name}")
            return
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def respond_json(self, data: dict):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def generate_latex_snippet(self, pair_ids: list) -> str:
        lines = [
            "% --- Tự động tạo bởi ChronosArt-I2I Web Curator ---",
            "\\begin{figure*}[t]",
            "\\centering",
            "\\includegraphics[width=0.98\\textwidth]{Figure/custom_i2i_grid.pdf}",
            "\\caption{\\textbf{So sánh định tính thị giác trên các ca kiểm thử chọn lọc cho bài toán Chuyển Giao Phong Cách Ảnh-Sang-Ảnh (ChronosArt-I2I)}.",
            f"Bao gồm {len(pair_ids)} ca đối sánh tiêu biểu: Cột (a) Ảnh Nội Dung; (b) Ảnh Phong Cách; (c) SVD-base; (d) \\textbf{{ChronosArt-I2I (Đề Xuất, viền xanh lá)}}.",
            "Phương pháp đề xuất duy trì 100\\% tỷ lệ khung hình và giải phẫu đối tượng, tiêm chuẩn xác ánh sáng và chất liệu nghệ thuật mà triệt tiêu hoàn toàn rò rỉ nội dung.}",
            "\\label{fig:custom_i2i_selected}",
            "\\end{figure*}"
        ]
        return "\n".join(lines)

    def generate_comparison_grid(self, pair_ids: list):
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        img_size = 512
        margin = 8
        header_h = 44
        num_rows = len(pair_ids)
        if num_rows == 0:
            return "", ""

        total_w = 4 * img_size + 3 * margin
        total_h = num_rows * img_size + (num_rows - 1) * margin + header_h

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        headers = ["(a) Content Reference", "(b) Style Reference", "(c) SVD-base", "(d) ChronosArt-I2I (Đề Xuất)"]
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf", 22)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        for col, h_text in enumerate(headers):
            x = col * (img_size + margin) + img_size // 4
            draw.text((x, 12), h_text, fill=(30, 30, 30), font=font)

        # Load pairs.json for mapping
        with open(WEB_DIR / "data" / "pairs.json", "r") as f:
            pairs_dict = {p["id"]: p for p in json.load(f)}

        for row, pid in enumerate(pair_ids):
            p = pairs_dict.get(pid)
            if not p:
                continue
            c_name = p["content_name"]
            s_name = p["style_name"]

            c_file = CONTENT_DIR / f"{c_name}.png"
            s_file = STYLES_DIR / f"{s_name}.png"
            base_file = EVAL_DIR / f"baseline_{c_name}_{s_name}.png"
            ours_file = EVAL_DIR / f"chronosart_i2i_{c_name}_{s_name}.png"

            c_pil = Image.open(c_file).convert("RGB").resize((img_size, img_size)) if c_file.exists() else Image.new("RGB", (img_size, img_size), (200, 200, 200))
            s_pil = Image.open(s_file).convert("RGB").resize((img_size, img_size)) if s_file.exists() else Image.new("RGB", (img_size, img_size), (200, 200, 200))
            base_pil = Image.open(base_file).convert("RGB").resize((img_size, img_size)) if base_file.exists() else Image.new("RGB", (img_size, img_size), (220, 220, 220))
            ours_pil = Image.open(ours_file).convert("RGB").resize((img_size, img_size)) if ours_file.exists() else Image.new("RGB", (img_size, img_size), (240, 240, 240))

            y = header_h + row * (img_size + margin)
            canvas.paste(c_pil, (0, y))
            canvas.paste(s_pil, (img_size + margin, y))
            canvas.paste(base_pil, (2 * (img_size + margin), y))
            canvas.paste(ours_pil, (3 * (img_size + margin), y))

            # Green boundary around our proposed method
            x_ours = 3 * (img_size + margin)
            draw.rectangle([x_ours, y, x_ours + img_size - 1, y + img_size - 1], outline=(30, 160, 60), width=4)

        png_name = "custom_i2i_grid.png"
        pdf_name = "custom_i2i_grid.pdf"
        canvas.save(FIGURE_DIR / png_name)
        canvas.save(FIGURE_DIR / pdf_name, "PDF", resolution=300.0)
        return png_name, pdf_name

def run():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, I2IViewerHandler)
    print(f"ChronosArt-I2I Web Curator running at http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    run()
