import re
import json
from pyvi import ViTokenizer

re_thuchientheo = re.compile(
    r"((((được\s)?thực hiện theo qu[iy] định tại\s|hướng dẫn tại\s|theo qu[iy] định tại\s|(được\s)?thực hiện theo\s|theo qu[iy] định tại\s|theo nội dung qu[yi] định tại\s|quy[iy] định tại|theo\s)(các\s)?)?|tại\s(các\s)?)(khoản(\ssố)?\s(\d+\,\s)*\d+|điều(\ssố)?\s(\d+\,\s)*\d+|điểm\s(([a-z]|đ)\,\s)*([a-z]|đ)\b|chương(\ssố)?\s(\d+\,\s)*\d+)((\s|\,\s|\s\,\s|\svà\s)(khoản(\ssố)?\s(\d+\,\s)*\d+|điều(\ssố)?\s(\d+\,\s)*\d+|điểm\s(([a-z]|đ)\,\s)*([a-z]|đ)\b|chương(\ssố)?\s(\d+\,\s)*\d+))*(\s(điều này|thông tư này|nghị quyết này|quyết định này|nghị định này|văn bản này|quyết định này))?"
)
re_thongtuso = re.compile(
    r"(thông tư liên tịch|thông tư|nghị quyết|quyết định|nghị định|văn bản)\s(số\s)?(([a-z0-9]|đ|\-)+\/([a-z0-9]|đ|\-|\/)*)"
)
re_ngay = re.compile(r"ngày\s\d+\/\d+\/\d+\b|ngày\s\d+tháng\d+năm\d+")
re_thang_nam = re.compile(r"tháng\s\d+\/\d+|tháng\s\d+|năm\s\d+")
re_chuong = re.compile(
    r"chương\s(iii|ii|iv|ix|viii|vii|vi|xi|xii|xiii|xiv|xix|xviii|xvii|xvi|xv|xx|v|x|i|xxiii|xxii|xxi|xxiv|xxviii|xxvii|xxvi|xxv|xxix|xxx)\b"
)
re_dieu = re.compile(r"Điều\s\d+\.")
END_PHRASES = [
    "có đúng không",
    "đúng không",
    "được không",
    "hay không",
    "được hiểu thế nào",
    "được quy định cụ thể là gì",
    "được quy định như thế nào",
    "được quy định thế nào",
    "được quy định như nào",
    "trong trường hợp như nào",
    "trong trường hợp như thế nào",
    "trong trường hợp nào",
    "trong những trường hợp nào",
    "được hiểu như thế nào",
    "được hiểu như nào",
    "như thế nào",
    "thế nào",
    "như nào",
    "là gì",
    "là ai",
    "là bao nhiêu",
    "bao nhiêu",
    "trước bao lâu",
    "là bao lâu",
    "bao lâu",
    "bao gồm gì",
    "không",
    "bao gồm những gì",
    "vào thời điểm nào",
    "gồm những giấy tờ gì",
    "những yêu cầu nào",
]

def remove_dieu_number(text):
    text = re_thuchientheo.sub(" ", text)
    text = re_thongtuso.sub(" ", text)
    text = re_ngay.sub(" ", text)
    text = re_thang_nam.sub(" ", text)
    text = re_chuong.sub(" ", text)
    text = re_dieu.sub(" ", text)
    return " ".join(text.split())


def remove_other_number_by_zero(text):
    for digit in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        text = text.replace(digit, "0")
    return text


def remove_punct(text):
    text = text.replace(";", ",").replace(":", ".").replace("“", " ").replace("”", " ")
    text = "".join(
        [
            c
            if c.isalpha() or c.isdigit() or c in [" ", ",", "(", ")", ".", "/", "-"]
            else " "
            for c in text
        ]
    )
    text = " ".join(text.split())
    text = text.replace("\xa0", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_khoan(khoan):
    khoan = khoan.lower()
    matched = re.match(r"^\d+\.(\d+\.?)?\s", khoan)  # 1. 2.2. 2.2
    if matched is not None:
        khoan = khoan[matched.span()[1]:].strip()

    else:
        matched2 = re.match(r"^[\wđ]\)\s", khoan)
        if matched2 is not None:
            khoan = khoan[matched2.span()[1]:].strip()

    khoan = remove_dieu_number(khoan)
    khoan = remove_other_number_by_zero(khoan)
    khoan = remove_punct(khoan)
    return " ".join(khoan.split())

def preprocess_question(q, remove_end_phrase=True):
    q = q.lower()
    q = remove_dieu_number(q)
    q = "".join([c if c.isalpha() or c.isdigit() or c == " " else " " for c in q])
    q = remove_punct(q)
    if remove_end_phrase:
        for phrase in END_PHRASES:
            if q.endswith(phrase):
                q = q[: -len(phrase)]
                break

    return q.strip()

def tokenize_text(text):
    return ViTokenizer.tokenize(text)

punc = """!"#$%&'()*+,-./:;<=>?@[\]^`{|}~"""  # noqa: W605
table = str.maketrans("", "", punc)


def clean_text(text):
    words = text.lower().split()
    result = [w.translate(table) for w in words]
    stripped = " ".join(result)
    result = " ".join(stripped.split())
    return result