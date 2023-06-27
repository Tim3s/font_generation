import os
import argparse
from PIL import Image,ImageDraw,ImageFont
 
# 이미지로 출력할 폰트 지정
parser = argparse.ArgumentParser(description='Convert Font File to Images')
parser.add_argument("-p", "--path", type=str, default="./fonts/NanumGothic.otf", help="path to font file(otf or ttf)")
args = parser.parse_args()
data_loc = args.path
processed_data_loc = os.path.join(".", "dataset", os.path.splitext(os.path.basename(data_loc))[0])
if not os.path.isdir("./dataset"):
    os.mkdir("./dataset")
if not os.path.isdir(processed_data_loc):
    os.mkdir(processed_data_loc)
font = ImageFont.truetype(data_loc, 300)
 
# 이미지 사이즈 지정
text_width = 300
text_height = 300
for idx in range(44032, 55204):
    draw_text = chr(idx) 
    # 이미지 객체 생성 (배경 검정)
    canvas = Image.new('L', (text_width, text_height), "black")
    
    # 가운데에 그리기 (폰트 색: 하양)
    draw = ImageDraw.Draw(canvas)
    l, t, r, b = font.getbbox(draw_text)
    draw.text(((text_width-r-l)/2.0,(text_height-b-t)/2.0), draw_text, 'white', font)
    
    # png로 저장 및 출력해서 보기
    canvas.save(os.path.join(processed_data_loc, draw_text+'.png'), "PNG")