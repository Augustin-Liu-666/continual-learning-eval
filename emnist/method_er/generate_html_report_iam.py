# File: emnist/method_er/generate_html_report_iam.py

import os

def generate_html_report(title, subtitle, log_path, output_path):
    if not os.path.exists(log_path):
        print("❌ The log file was not found; the HTML report cannot be generated.")
        return

    with open(log_path, "r") as f:
        log_lines = f.read().strip().split("\n")

    # Build the HTML content.
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='utf-8'>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 40px;
                background-color: #f9f9f9;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #ccc;
            }}
            h2 {{
                color: #444;
            }}
            pre {{
                background-color: #f0f0f0;
                padding: 16px;
                border-left: 5px solid #007acc;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <h2>{subtitle}</h2>

        <h3>🧾 Training log</h3>
        <pre>
{chr(10).join(log_lines)}
        </pre>

        <p>📄 This report contains the output of the IAM single-task T5 experiment.</p>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html)

    print(f"📄 HTML report generated: {output_path}")
