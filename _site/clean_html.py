import re
import sys


def clean_html(html_content):
    html_content = re.sub(
        r"<!DOCTYPE[^>]*>\s*", "", html_content, flags=re.IGNORECASE
    )
    # html_content = re.sub(
    #     r"<style.*?>.*?</style>",
    #     "",
    #     html_content,
    #     flags=re.DOTALL | re.IGNORECASE,
    # )
    return html_content


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as f:
        html = f.read()

    cleaned_html = clean_html(html)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_html)
