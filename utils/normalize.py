import re
import sys
import unicodedata
import string
from pyvi import ViTokenizer

# Hàm chuẩn hoá các từ tiếng việt có dấu
def nomarlize_text(string):
    for k, v in replace_list1.items():
        string = string.replace(k, v)
    return string
replace_list1 = {
    # trường hợp đặt sai dấu
    'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
    'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
    'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ',
    'ủa':'uả','úa':'uá', 'ùa':'uà','ũan':'uãn','ụa':'uạ'}
    

# hàm xoá dấu tiếng việt bằng unicodedata
python_version = sys.version_info[0]
def no_marks(s):
    if python_version < 3:
        s = s.decode('utf-8')
        s = re.sub(unichr(272), 'D', s)
        s = re.sub(unichr(273), 'd', s)
    else:
        s = re.sub('\u0110', 'D', s)
        s = re.sub('\u0111', 'd', s)
    if python_version < 3:
        return unicodedata.normalize('NFKD', unicode(s)).encode('ASCII', 'ignore').decode("utf-8") 
    else:
        return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode("utf-8") 

# Hàm chuẩn hoá các cách viết teencode ở trên mạng xã hội
# Chuẩn hoá các emoj cảm xúc thành positive và negative
def nomarlize_teen_code(text):
    text = text.lower()

    # chuyển các dấu chấm thành dấu cách  , hàm maketrans() để chuyển dấu chấm về dấu cách
    # chuyen punctuation thành space 
    # mục đích của dòng này để xóa các dấu '.' , ',' , '????' , '!!!'
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    #Remove các ký tự kéo dài: vd: đẹppppppp thành đẹP ( nó bị lỗi chuyển thành chữ hoa , do đó cần chuyển lại về chữ thường)
    text = re.sub(r'(a-z])\1+', lambda m: m.group(1), text, flags=re.IGNORECASE)
    
    for k, v in replace_list2.items():
        text = text.replace(k, v)
    
    text = ViTokenizer.tokenize(text)
    text = re.sub(r'( )\1+', lambda m: m.group(1), text, flags=re.IGNORECASE)
    return text
# từ điển chuẩn hoá
replace_list2 = {
    # trường hợp đặt sai dấu
    'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
    'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
    'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ',
    'ủa':'uả','úa':'uá', 'ùa':'uà','ũan':'uãn','ụa':'uạ',
    # trường hợp có dấu khác encode
    'àk': 'à', 'aˋ': 'à', 'a´': 'á','ả': 'ả','a˜':'ã',
    'ăˋ':'ằ', 'ă´': 'ắ','ắ': 'ắ',
    'ầ': 'ầ','âˋ':'ầ', 'â´': 'ấ','ấ': 'ấ', 'ẩ': 'ẩ', 'ẫ': 'ẫ',
    'oˋ':'ò', 'o´': 'ó', 'ỏ': 'ỏ','o˜':'õ',
    'ồ': 'ồ','ôˋ':'ồ','ố': 'ố', 'ô´': 'ố', 'ổ': 'ổ','ỗ': 'ỗ','ô˜':'ỗ',
    'ơˋ':'ờ', 'ơ´': 'ớ', 'ở': 'ở',
    'eˋ':'è', 'e´': 'é', 'ẻ': 'ẻ','e˜': 'ẽ',
    'êˋ':'ề', 'ê´':'ế','ề': 'ề', 'ế': 'ế','ễ': 'ễ','ê˜':'ễ',
    'uˋ':'ù', 'u´':'ú','ủ':'ủ','u˜':'ũ',
    'ưˋ':'ừ', 'ư´':'ứ','ử':'ử', 'ư˜':'ữ',
    'iˋ':'ì','i´':'í', 'ỉ':'ỉ',
    'yˋ':'ỳ', 'y´':'ý', 'y˜':'ỹ',
    # trường hợp đẹp từ đẹp quáaa không xoá được áa
    'áa':'á',
    #Quy các icon về 2 loại emoj: Tích cực hoặc tiêu cực
    "👹": "nagative", "👻": "positive", "💃": "positive",'🤙': ' positive ', '👍': ' positive ',
    "💄": "positive", "💎": "positive", "💩": "positive","😕": "nagative", "😱": "nagative", "😸": "positive",
    "😾": "nagative", "🚫": "nagative",  "🤬": "nagative","🧚": "positive", "🧡": "positive",'🐶':' positive ',
    '👎': ' nagative ', '😣': ' nagative ','✨': ' positive ', '❣': ' positive ','☀': ' positive ',
    '♥': ' positive ', '🤩': ' positive ', 'like': ' positive ', '💌': ' positive ',
    '🤣': ' positive ', '🖤': ' positive ', '🤤': ' positive ', ':(': ' nagative ', '😢': ' nagative ',
    '❤': ' positive ', '😍': ' positive ', '😘': ' positive ', '😪': ' nagative ', '😊': ' positive ',
    '?': ' ? ', '😁': ' positive ', '💖': ' positive ', '😟': ' nagative ', '😭': ' nagative ',
    '💯': ' positive ', '💗': ' positive ', '♡': ' positive ', '💜': ' positive ', '🤗': ' positive ',
    '^^': ' positive ', '😨': ' nagative ', '☺': ' positive ', '💋': ' positive ', '👌': ' positive ',
    '😖': ' nagative ', '😀': ' positive ', ':((': ' nagative ', '😡': ' nagative ', '😠': ' nagative ',
    '😒': ' nagative ', '🙂': ' positive ', '😏': ' nagative ', '😝': ' positive ', '😄': ' positive ',
    '😙': ' positive ', '😤': ' nagative ', '😎': ' positive ', '😆': ' positive ', '💚': ' positive ',
    '✌': ' positive ', '💕': ' positive ', '😞': ' nagative ', '😓': ' nagative ', '️🆗️': ' positive ',
    '😉': ' positive ', '😂': ' positive ', ':v': '  positive ', '=))': '  positive ', '😋': ' positive ',
    '💓': ' positive ', '😐': ' nagative ', ':3': ' positive ', '😫': ' nagative ', '😥': ' nagative ',
    '😃': ' positive ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' positive ', '🤝': ' positive ', '🎈': ' positive ',
    '😗': ' positive ', '🤔': ' nagative ', '😑': ' nagative ', '🔥': ' nagative ', '🙏': ' nagative ',
    '🆗': ' positive ', '😻': ' positive ', '💙': ' positive ', '💟': ' positive ',
    '😚': ' positive ', '❌': ' nagative ', '👏': ' positive ', ';)': ' positive ', '<3': ' positive ',
    '🌝': ' positive ',  '🌷': ' positive ', '🌸': ' positive ', '🌺': ' positive ',
    '🌼': ' positive ', '🍓': ' positive ', '🐅': ' positive ', '🐾': ' positive ', '👉': ' positive ',
    '💐': ' positive ', '💞': ' positive ', '💥': ' positive ', '💪': ' positive ',
    '💰': ' positive ',  '😇': ' positive ', '😛': ' positive ', '😜': ' positive ',
    '🙃': ' positive ', '🤑': ' positive ', '🤪': ' positive ','☹': ' nagative ',  '💀': ' nagative ',
    '😔': ' nagative ', '😧': ' nagative ', '😩': ' nagative ', '😰': ' nagative ', '😳': ' nagative ',
    '😵': ' nagative ', '😶': ' nagative ', '🙁': ' nagative ',
    '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' positive ',
    ':))': '  positive ', ':)': ' positive ',
    #dưới 3* quy về 1*, trên 3* quy về 5*
    '6 sao': ' 5star ','6 star': ' 5star ', '5star': ' 5star ','5 sao': ' 5star ','5sao': ' 5star ',
    'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ','2 sao':' 1star ','2sao':' 1star ',
    '2 starstar':' 1star ','1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ',
    #Chuẩn hóa 1 số sentiment words/English words
    'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
    'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
    ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ', 'hong':'không',
    'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',
    ' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
    'he he': ' positive ','hehe': ' positive ','hihi': ' positive ', 'haha': ' positive ', 'hjhj': ' positive ',
    ' lol ': ' nagative ',' cc ': ' nagative ','cute': u' dễ thương ','huhu': ' nagative ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
    ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
    'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' positive ', 'store': u' cửa hàng ',
    'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
    'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
    'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
    'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',
    'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
    'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
    'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
    'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
    'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',
    'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' positive ',
    ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback ',
    #dưới 3* quy về 1*, trên 3* quy về 5*
    '6 sao': ' 5star ','6 star': ' 5star ', '5star': ' 5star ','5 sao': ' 5star ','5sao': ' 5star ',
    'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ','2 sao':' 1star ','2sao':' 1star ',
    '2 starstar':' 1star ','1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ',
    u'"':u' ', u'️':u'', '🏻':'', "'":"",
    }
#########################################################################################################