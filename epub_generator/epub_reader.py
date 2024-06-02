import html_text
from ebooklib import epub


def read_epub(book_path: str):
    book = epub.read_epub(book_path)
    items = book.get_items()
    chapters = []
    for item in items:
        contents = item.get_content()
        try:
            contents = contents.decode('utf-8')
            contents = html_text.extract_text(contents)
            contents = contents.split('\n')
            contents = [content.strip() for content in contents if content and content != "\n"]
            if contents:
                chapters.append(contents)
        except Exception as e:
            pass
    return chapters


if __name__ == '__main__':
    book = read_epub("冬牧场.epub")
