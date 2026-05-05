import re
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont


CANVAS_WIDTH = 1024
CANVAS_HEIGHT = 1024
PADDING = 40
SPACING = 30
OUTPUT_DIR = "./zebra_canvases"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def get_font(size):
    candidates =[
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf"   
    ]
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

def get_image_from_sample(sample, key):

    val = sample.get(key)
    if isinstance(val, list):
        return val[0] if len(val) > 0 else None
    return val
def get_font_path():
    candidates =[
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
    ]
    for c in candidates:
        if os.path.exists(c): return c
    return None
font_path = get_font_path()
if not font_path:
    raise FileNotFoundError("找不到标准的等宽 TTF 字体文件，请安装。")
FONT_CACHE = {}


# ==========================================================
# 工具函数 1：在给定的矩形框 (x0, y0, max_w, max_h) 内自适应渲染纯文本
# (用于上下排版和左右排版，这种排版不需要处理异形环绕，效率极高)
# ==========================================================
def draw_text_in_box(draw, text, box_rect, W, H):
    x0, y0, max_w, max_h = box_rect
    best_f, best_lines, best_g, best_font = 14,[], 0, None
    
    # 动态寻找最优字体大小
    for f_size in range(80, 13, -1):
        if f_size not in FONT_CACHE:
            FONT_CACHE[f_size] = get_font(f_size)
        font = FONT_CACHE[f_size]
        
        char_width = font.getlength('x') if hasattr(font, 'getlength') else font.getbbox('x')[2]
        m = int(max_w / char_width)
        if m <= 0: continue
            
        lines = textwrap.wrap(text, width=m, break_long_words=True)
        L = len(lines)
        g = f_size / 4  # 行距
        total_text_h = L * (f_size + g)
        
        if total_text_h <= max_h:
            best_f, best_lines, best_g, best_font = f_size, lines, g, font
            break

    # 极端降级处理
    if not best_font:
        best_f = 14
        best_font = get_font(best_f)
        char_width = best_font.getlength('x') if hasattr(best_font, 'getlength') else best_font.getbbox('x')[2]
        m = int(max_w / char_width)
        best_lines = textwrap.wrap(text, width=m, break_long_words=True)
        best_g = best_f / 4
        
    curr_y = y0
    for line in best_lines:
        draw.text((x0, curr_y), line, font=best_font, fill='black')
        curr_y += best_f + best_g


# ==========================================================
# 策略 A：上下排版 (Top-Bottom)
# 适用场景：小图 (<512x512) 放大，横宽图 (W>1024) 缩小
# ==========================================================
def render_top_bottom(text, img_obj, W=1024, H=1024, p=40, gap=30):
    # 下半部分最多占 50% 空间留给图像
    max_img_w = W - 2 * p
    max_img_h = int((H - 2 * p - gap) * 0.5)
    
    # 比例自适应（兼容了小图放大和大图缩小的逻辑）
    ratio = min(max_img_w / float(img_obj.width), max_img_h / float(img_obj.height))
    new_w, new_h = int(img_obj.width * ratio), int(img_obj.height * ratio)
    img_obj = img_obj.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 图片底部居中
    img_x = (W - new_w) // 2
    img_y = H - p - new_h
    
    # 文本放上半部分
    text_box = (p, p, W - 2*p, img_y - gap - p)
    
    canvas = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(canvas)
    canvas.paste(img_obj, (img_x, img_y))
    draw.rectangle([img_x-1, img_y-1, img_x+new_w, img_y+new_h], outline="#CCCCCC", width=2)
    
    draw_text_in_box(draw, text, text_box, W, H)
    return canvas


# ==========================================================
# 策略 B：左右排版 (Left-Right)
# 适用场景：竖长图 (H>1024) 缩小
# ==========================================================
def render_left_right(text, img_obj, W=1024, H=1024, p=40, gap=30):
    # 左半部分图，右半部分文
    half_w = (W - 2*p - gap) // 2
    max_img_w = half_w
    max_img_h = H - 2 * p
    
    ratio = min(max_img_w / float(img_obj.width), max_img_h / float(img_obj.height))
    new_w, new_h = int(img_obj.width * ratio), int(img_obj.height * ratio)
    img_obj = img_obj.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 图片垂直居中，放在左侧
    img_x = p + (half_w - new_w) // 2
    img_y = (H - new_h) // 2
    
    # 文本放右半部分
    text_x = p + half_w + gap
    text_box = (text_x, p, W - text_x - p, H - 2*p)
    
    canvas = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(canvas)
    canvas.paste(img_obj, (img_x, img_y))
    draw.rectangle([img_x-1, img_y-1, img_x+new_w, img_y+new_h], outline="#CCCCCC", width=2)
    
    draw_text_in_box(draw, text, text_box, W, H)
    return canvas


# ==========================================================
# 策略 C：左下角文字环绕排版 (Bottom-Left Wrap) - 你的旧版本
# 适用场景：512~1024 的中等常规图片
# ==========================================================
def try_layout_text_adaptive(text, font, img_box, W, H, p, gap):
    img_x, img_y, img_w, img_h = img_box
    line_height = int(font.size * 1.4)
    
    def get_line_bounds(y):
        if y + line_height > img_y and y < img_y + img_h:
            start_x = img_x + img_w + gap
            max_w = W - start_x - p
            return start_x, max_w
        else:
            start_x = p
            max_w = W - 2 * p
            return start_x, max_w

    words =[]
    for token in text.split(' '):
        if '\n' in token:
            parts = token.split('\n')
            for i, part in enumerate(parts):
                if part: words.append(part)
                if i < len(parts) - 1: words.append('\n')
        else:
            if token: words.append(token)

    lines_to_draw =[]
    current_y = p
    current_x, current_max_w = get_line_bounds(current_y)
    current_line =[]
    
    for word in words:
        if word == '\n':
            lines_to_draw.append((current_x, current_y, " ".join(current_line)))
            current_y += line_height
            if current_y + line_height > H - p: return False,[]
            current_x, current_max_w = get_line_bounds(current_y)
            current_line =[]
            continue

        test_line = " ".join(current_line + [word]) if current_line else word
        length = font.getlength(test_line) if hasattr(font, 'getlength') else len(test_line)*font.size*0.6
        
        if length <= current_max_w:
            current_line.append(word)
        else:
            if current_line:
                lines_to_draw.append((current_x, current_y, " ".join(current_line)))
                current_y += line_height
                if current_y + line_height > H - p: return False,[]
                current_x, current_max_w = get_line_bounds(current_y)
                current_line = [word]
            else:
                lines_to_draw.append((current_x, current_y, word))
                current_y += line_height
                if current_y + line_height > H - p: return False,[]
                current_x, current_max_w = get_line_bounds(current_y)
            
    if current_line:
        lines_to_draw.append((current_x, current_y, " ".join(current_line)))
        if current_y + line_height > H - p: return False,[]
            
    return True, lines_to_draw

def render_bottom_left_wrap(text, img_obj, W=1024, H=1024, p=40, gap=30):
    max_img_w, max_img_h = int(W * 0.50), int(H * 0.50)
    ratio = min(max_img_w / float(img_obj.width), max_img_h / float(img_obj.height))
    
    if ratio < 1.0:
        new_w, new_h = int(img_obj.width * ratio), int(img_obj.height * ratio)
        img_obj = img_obj.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    img_x = p
    img_y = H - p - img_obj.height
    img_box = (img_x, img_y, img_obj.width, img_obj.height)

    best_lines, best_font =[], None
    for f_size in range(60, 13, -1):
        if f_size not in FONT_CACHE:
            FONT_CACHE[f_size] = get_font(f_size)
        font = FONT_CACHE[f_size]
        success, lines = try_layout_text_adaptive(text, font, img_box, W, H, p, gap)
        if success:
            best_lines = lines
            best_font = font
            break
            
    if not best_font:
        best_font = get_font(14)
        _, best_lines = try_layout_text_adaptive(text, best_font, img_box, W, H, p, gap)

    canvas = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(canvas)
    
    canvas.paste(img_obj, (img_box[0], img_box[1]))
    draw.rectangle([img_box[0]-1, img_box[1]-1, img_box[0]+img_box[2], img_box[1]+img_box[3]], outline="#CCCCCC", width=1)
        
    for line_x, line_y, line_text in best_lines:
        draw.text((line_x, line_y), line_text, font=best_font, fill='black')
        
    return canvas


# ==========================================================
# 🌟 核心路由器：根据宽高智能分发渲染策略
# ==========================================================
def render_wrapped_canvas_adaptive(text, img_obj, W=CANVAS_WIDTH, H=CANVAS_HEIGHT, p=PADDING, gap=SPACING):
    orig_w, orig_h = img_obj.width, img_obj.height
    
    # 策略 1：小图（宽高均小于512） -> 放大置于底部
    if orig_w < 512 and orig_h < 512:
        # print(f"  [Router] 采用上下排版 (小图放大) -> {orig_w}x{orig_h}")
        return render_top_bottom(text, img_obj, W, H, p, gap)
        
    # 策略 2/3：遇到长/大图 (>1024)
    elif orig_h > 1024 or orig_w > 1024:
        if orig_h > orig_w:
            # 策略 2：竖长图 -> 左右分栏
            # print(f"  [Router] 采用左右排版 (竖长图) -> {orig_w}x{orig_h}")
            return render_left_right(text, img_obj, W, H, p, gap)
        else:
            # 策略 3：横宽图 -> 上下分栏
            # print(f"  [Router] 采用上下排版 (横宽图) -> {orig_w}x{orig_h}")
            return render_top_bottom(text, img_obj, W, H, p, gap)
            
    # 策略 4：其它常规图像（如 800x600 等） -> 自适应左下角环绕
    else:
        # print(f"  [Router] 采用左下角环绕排版 (中等图) -> {orig_w}x{orig_h}")
        return render_bottom_left_wrap(text, img_obj, W, H, p, gap)


def render_cot_to_image(cot_text, W=1024, H=1024, p=20, f_max=100, f_min=10):
    best_f, best_lines, best_g, best_font = f_min,[], 0, None
    
    for f in range(f_max, f_min - 1, -1):
        if f not in FONT_CACHE:
            FONT_CACHE[f] = ImageFont.truetype(font_path, f)
        font = FONT_CACHE[f]
        
        char_width = font.getlength('x') if hasattr(font, 'getlength') else font.getbbox('x')[2]
        m = int((W - 2 * p) / char_width)
        if m <= 0: continue
            
        lines = textwrap.wrap(cot_text, width=m, break_long_words=True)
        L = len(lines)
        g = f / 4  
        h = L * (f + g) + 2 * p
        
        if h <= H:
            best_f, best_lines, best_g, best_font = f, lines, g, font
            break

    img = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(img)
    y_text = p
    for line in best_lines:
        draw.text((p, y_text), line, font=best_font, fill='black')
        y_text += best_f + best_g
    
    return img

def try_layout_text(text, font, img_box, W, H, p, gap):
    img_x, img_y, img_w, img_h = img_box
    line_height = int(font.size * 1.4)
    
    def get_line_bounds(y):
        if y + line_height > img_y and y < img_y + img_h:
            start_x = img_x + img_w + gap
            max_w = W - start_x - p
            return start_x, max_w
        else:
            start_x = p
            max_w = W - 2 * p
            return start_x, max_w

    words =[]
    for token in text.split(' '):
        if '\n' in token:
            parts = token.split('\n')
            for i, part in enumerate(parts):
                if part: words.append(part)
                if i < len(parts) - 1: words.append('\n')
        else:
            if token: words.append(token)

    lines_to_draw =[]
    current_y = p
    current_x, current_max_w = get_line_bounds(current_y)
    current_line =[]
    
    for word in words:
        if word == '\n':
            lines_to_draw.append((current_x, current_y, " ".join(current_line)))
            current_y += line_height
            if current_y + line_height > H - p: return False,[]
            current_x, current_max_w = get_line_bounds(current_y)
            current_line = []
            continue

        test_line = " ".join(current_line +[word]) if current_line else word
        length = font.getlength(test_line) if hasattr(font, 'getlength') else len(test_line)*font.size*0.6
        
        if length <= current_max_w:
            current_line.append(word)
        else:
            if current_line:
                lines_to_draw.append((current_x, current_y, " ".join(current_line)))
                current_y += line_height
                if current_y + line_height > H - p: return False,[]
                current_x, current_max_w = get_line_bounds(current_y)
                current_line = [word]
            else:
                lines_to_draw.append((current_x, current_y, word))
                current_y += line_height
                if current_y + line_height > H - p: return False,[]
                current_x, current_max_w = get_line_bounds(current_y)
            
    if current_line:
        lines_to_draw.append((current_x, current_y, " ".join(current_line)))
        if current_y + line_height > H - p: return False,[]
            
    return True, lines_to_draw

def render_wrapped_canvas(text, img_obj, W=CANVAS_WIDTH, H=CANVAS_HEIGHT, p=PADDING, gap=SPACING):
    max_img_w, max_img_h = int(W * 0.50), int(H * 0.50)
    ratio = min(max_img_w / float(img_obj.width), max_img_h / float(img_obj.height))
    
    if ratio < 1.0:
        new_w, new_h = int(img_obj.width * ratio), int(img_obj.height * ratio)
        img_obj = img_obj.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    img_x = p
    img_y = H - p - img_obj.height
    img_box = (img_x, img_y, img_obj.width, img_obj.height)

    best_lines, best_font =[], None
    for f_size in range(60, 13, -1):
        font = get_font(f_size)
        success, lines = try_layout_text(text, font, img_box, W, H, p, gap)
        if success:
            best_lines = lines
            best_font = font
            break
            
    if not best_font:
        print("    [警告] 文本过多，强制使用 14px 且可能会被截断。")
        best_font = get_font(14)
        _, best_lines = try_layout_text(text, best_font, img_box, W, H, p, gap)

    canvas = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(canvas)
    
    canvas.paste(img_obj, (img_box[0], img_box[1]))
    draw.rectangle([img_box[0]-1, img_box[1]-1, img_box[0]+img_box[2], img_box[1]+img_box[3]], outline="#CCCCCC", width=1)
        
    for line_x, line_y, line_text in best_lines:
        draw.text((line_x, line_y), line_text, font=best_font, fill='black')
        
    return canvas
def render_wrapped_canvas4updown(text, img_obj, W=CANVAS_WIDTH, H=CANVAS_HEIGHT, p=PADDING, gap=SPACING):
    """
    修改为【上下排版策略】：
    - 下半部分（默认最多占据50%高度）用于居中放置辅助图片
    - 上半部分用于纯文本渲染（自适应字体大小）
    """
    # 1. 确定下半部分图像的最大可用空间 (例如占最多 50% 高度)
    max_img_w = W - 2 * p
    max_img_h = int((H - 2 * p - gap) * 0.5) 
    
    # 计算缩放比例，保证图像不超出宽/高限制，且保持原比例
    ratio = min(max_img_w / float(img_obj.width), max_img_h / float(img_obj.height))
    
    new_w = int(img_obj.width * ratio)
    new_h = int(img_obj.height * ratio)
    # 使用 LANCZOS 确保缩放后的图像清晰
    img_obj = img_obj.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 2. 计算图像的位置：底部对齐，水平居中
    img_x = (W - new_w) // 2
    img_y = H - p - new_h
    
    # 3. 确定上半部分文本的可用空间
    max_text_w = W - 2 * p
    max_text_h = img_y - gap - p  # 文本最多可以写到图片上方 gap 距离处
    
    best_f, best_lines, best_g, best_font = 14,[], 0, None
    
    # 4. 动态寻找最优字体大小，以完美填满上半部分
    for f_size in range(80, 13, -1):
        if f_size not in FONT_CACHE:
            FONT_CACHE[f_size] = get_font(f_size)
        font = FONT_CACHE[f_size]
        
        # 估算每行可以放多少字符 (char_width 取 'x' 的宽度做近似)
        char_width = font.getlength('x') if hasattr(font, 'getlength') else font.getbbox('x')[2]
        m = int(max_text_w / char_width)
        if m <= 0: 
            continue
        
        # 使用 textwrap 自动对文本进行自然换行
        lines = textwrap.wrap(text, width=m, break_long_words=True)
        L = len(lines)
        g = f_size / 4  # 行间距
        total_text_h = L * (f_size + g)
        
        # 如果文本总高度小于可用高度，说明这个字体合适
        if total_text_h <= max_text_h:
            best_f, best_lines, best_g, best_font = f_size, lines, g, font
            break
            
    # 如果文本实在太多（连 14px 都塞不下），强制使用 14px 并打出警告
    if not best_font:
        print("    [警告] 文本过多，强制使用 14px，超出的文本将被截断。")
        best_f = 14
        best_font = get_font(best_f)
        char_width = best_font.getlength('x') if hasattr(best_font, 'getlength') else best_font.getbbox('x')[2]
        m = int(max_text_w / char_width)
        best_lines = textwrap.wrap(text, width=m, break_long_words=True)
        best_g = best_f / 4
    
    # 5. 开始正式绘制 Canvas
    canvas = Image.new('RGB', (W, H), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # 贴图
    canvas.paste(img_obj, (img_x, img_y))
    # 在图片外围画一圈细灰色边框，让辅助图显得更清晰有界限
    draw.rectangle([img_x-1, img_y-1, img_x+new_w, img_y+new_h], outline="#CCCCCC", width=2)
    
    # 写字（从最上方 padding 处往下写）
    y_text = p
    for line in best_lines:
        draw.text((p, y_text), line, font=best_font, fill='black')
        y_text += best_f + best_g
        
    return canvas

def process_single_sample(sample, sample_index=0):
    #print(f"\n{'='*50}\n开始处理第 {sample_index} 个真实样本...\n{'='*50}")
    
    question_raw = sample['Question']
    trace_raw = sample['Text Reasoning Trace']
    final_answer = sample['Final Answer']
    
    # 1. 提取 Question 和 Problem Image
    prob_img_match = re.search(r'<image_start>\[(.*?)\]<image_end>', question_raw)
    if prob_img_match:
        p_img_key = prob_img_match.group(1)
        problem_image = get_image_from_sample(sample, p_img_key)
        
        question_text = re.sub(r'<image_start>\[.*?\]<image_end>', '', question_raw).strip()
    else:
        question_text = question_raw
        problem_image = None

    print(f"提取问题: {question_text[:50]}...")
    if problem_image:
        prob_img_path = os.path.join(OUTPUT_DIR, f"sample_{sample_index}_problem_image.png")
        problem_image.save(prob_img_path)
        print(f"  [保存] 提取到原问题图像: {prob_img_path}")

    # 2. 分切 THOUGHT 链
    raw_thoughts = re.split(r'(?=THOUGHT \d+:)', trace_raw)
    raw_thoughts =[t.strip() for t in raw_thoughts if t.strip()]
    
    canvases =[]
    
    for i, thought_content in enumerate(raw_thoughts):
        print(f"\n  >> 渲染 THOUGHT {i}:")
        
        # 查找当前 thought 中是否包含对应的推理图片
        img_match = re.search(r'<image_start>\[(.*?)\]<image_end>', thought_content)
        
        if img_match:
            img_key = img_match.group(1)
            # 剥离图片标记，保留纯净文本
            pure_text = re.sub(r'<image_start>\[.*?\]<image_end>', '', thought_content).strip()
            
            img_obj = get_image_from_sample(sample, img_key)
            if img_obj is not None:
                # 执行图文环绕渲染
                canvas = render_wrapped_canvas_adaptive(pure_text, img_obj)
                save_path = os.path.join(OUTPUT_DIR, f"sample_{sample_index}_thought_{i}_with_image.png")
                canvas.save(save_path)
                canvases.append(save_path)
                print(f"    [图文模式] 成功拼接: {save_path}")
            else:
                print(f"    [降级] 找不到对应的图像实体 '{img_key}'，回退到纯文本模式。")
                canvas = render_cot_to_image(pure_text)
                save_path = os.path.join(OUTPUT_DIR, f"sample_{sample_index}_thought_{i}_text_only.png")
                canvas.save(save_path)
                canvases.append(save_path)
        else:
            # 执行纯文本固定 1024 渲染
            canvas = render_cot_to_image(thought_content)
            save_path = os.path.join(OUTPUT_DIR, f"sample_{sample_index}_thought_{i}_text_only.png")
            canvas.save(save_path)
            canvases.append(save_path)
            print(f"    [纯文本模式] 固定1024x1024，动态字体已生成: {save_path}")

    print(f"\n 样本 {sample_index} 处理完成！")
    print(f"   最终答案(Ground Truth): {final_answer}")
    return canvases

if __name__ == "__main__":
    from datasets import load_dataset

    parquet_file = os.environ.get(
        "UNIVLR_ZEBRA_COT_PARQUET",
        "data/Zebra-CoT/3D Visual Reasoning - Multi-Hop Objects Counting/train-00000-of-00040.parquet",
    )
    
    print(f"Loading dataset from: {parquet_file}")
    try:
        dataset = load_dataset("parquet", data_files=parquet_file)
        
        # 提取真正的第一条样本进行验证
        first_sample = dataset["train"][3]
        
        process_single_sample(first_sample, sample_index=0)
        
    except Exception as e:
        print(f"数据集加载或处理失败，错误信息: {e}")
