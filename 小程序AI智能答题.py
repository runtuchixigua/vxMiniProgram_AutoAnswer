from AppKit import NSWorkspace
import Quartz
import io
import base64
import os
import time
from openai import OpenAI
import pyautogui
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

# 配置pyautogui安全设置
pyautogui.PAUSE = 1  # 每个操作后暂停1秒，防止操作过快
pyautogui.FAILSAFE = True  # 启用故障保护，移动鼠标到屏幕左上角可以中断操作

# OpenAI配置
OPENAI_CONFIG = {
    'api_key': '你滴apikey',
    'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'model': 'qwen-turbo-latest',
    'multimodal_model': 'qwen3-vl-plus',
    'embedding_model': 'text-embedding-v4',
    'rerank_model': 'gte-rerank-v2',
    'embedding_dimensions': 2048
}

# 创建OpenAI客户端实例
client = OpenAI(
    api_key=OPENAI_CONFIG['api_key'],
    base_url=OPENAI_CONFIG['base_url']
)

# macOS窗口捕获工具
# 使用Quartz框架获取窗口信息，这是在macOS上获取窗口信息的正确方式

def get_window_info():
    # 获取所有窗口信息，使用OnScreenOnly选项获取当前显示的窗口
    window_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    
    # 过滤掉的系统应用列表
    system_apps = {'Window Server', 'Dock', 'loginwindow', '通知中心', '程序坞', '控制中心', '聚焦', 'NotificationCenter'}
    
    filtered_windows = []
    
    for window in window_list:
        app_name = window.get('kCGWindowOwnerName', '未知应用')
        
        # 跳过系统应用窗口
        if app_name in system_apps:
            continue
        
        title = window.get('kCGWindowName', '无标题')
        
        # 获取窗口位置和大小
        bounds = window.get('kCGWindowBounds', {})
        left = bounds.get('X', 0)
        top = bounds.get('Y', 0)
        width = bounds.get('Width', 0)
        height = bounds.get('Height', 0)
        
        # 只添加有实际大小且不是太小的窗口（过滤掉图标和小控件）
        if width > 100 and height > 100:
            filtered_windows.append({
                'app_name': app_name,
                'title': title,
                'left': left,
                'top': top,
                'width': width,
                'height': height
            })
    
    return filtered_windows

def capture_window_screenshot(window_info):
    """
    捕获指定窗口的截图并保存到images文件夹
    """
    try:
        left = window_info['left']
        top = window_info['top']
        width = window_info['width']
        height = window_info['height']
        
        # 确保images文件夹存在
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            print(f"创建images文件夹: {images_dir}")
        
        print(f"尝试捕获区域: ({left}, {top}, {width}, {height})")
        
        # 使用更简单直接的方法：直接捕获指定窗口
        # 首先创建要捕获的区域
        capture_rect = Quartz.CGRectMake(left, top, width, height)
        
        # 直接捕获指定区域的屏幕内容
        cg_image = Quartz.CGWindowListCreateImage(
            capture_rect,
            Quartz.kCGWindowListOptionOnScreenOnly,
            Quartz.kCGNullWindowID,
            Quartz.kCGWindowImageDefault
        )
        
        if not cg_image:
            print("无法捕获窗口图像")
            return None
        
        # 为截图生成文件名
        app_name = window_info.get('app_name', 'unknown').replace(' ', '_')
        window_title = window_info.get('title', 'untitled').replace(' ', '_')
        timestamp = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{app_name}_{window_title}_{timestamp}.png"
        filepath = os.path.join(images_dir, filename)
        
        # 使用PIL库来处理图像（更可靠）
        try:
            from PIL import Image
            import io
            
            # 获取图像数据
            width = Quartz.CGImageGetWidth(cg_image)
            height = Quartz.CGImageGetHeight(cg_image)
            bits_per_component = Quartz.CGImageGetBitsPerComponent(cg_image)
            bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
            color_space = Quartz.CGImageGetColorSpace(cg_image)
            
            # 创建一个位图上下文来获取像素数据
            context = Quartz.CGBitmapContextCreate(
                None,
                width,
                height,
                bits_per_component,
                bytes_per_row,
                color_space,
                Quartz.CGImageGetBitmapInfo(cg_image)
            )
            
            if context:
                Quartz.CGContextDrawImage(context, Quartz.CGRectMake(0, 0, width, height), cg_image)
                
                # 获取像素数据
                data = Quartz.CGBitmapContextGetData(context)
                if data:
                    # 创建PIL图像
                    img = Image.frombytes(
                        'RGBA',
                        (width, height),
                        data,
                        'raw',
                        'RGBA',
                        0,
                        1
                    )
                    
                    # 保存图像到文件
                    img.save(filepath, format='PNG')
                    print(f"截图已保存到: {filepath}")
                    
                    # 转换为PNG并编码为base64
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    base64_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    print(f"截图成功 (PIL方法)，数据大小: {len(base64_encoded)} 字符")
                    return f"data:image/png;base64,{base64_encoded}"
        except ImportError:
            print("PIL库不可用，尝试使用替代方法")
        except Exception as pil_error:
            print(f"PIL处理失败: {str(pil_error)}")
        
        # 备用方法：使用NSBitmapImageRep直接转换
        try:
            from AppKit import NSBitmapImageRep, NSPNGFileType, NSData
            
            # 获取图像尺寸
            width = Quartz.CGImageGetWidth(cg_image)
            height = Quartz.CGImageGetHeight(cg_image)
            
            # 创建NSBitmapImageRep
            bitmap_rep = NSBitmapImageRep.alloc().initWithCGImage_(cg_image)
            
            if bitmap_rep:
                # 转换为PNG数据
                png_data = bitmap_rep.representationUsingType_properties_(NSPNGFileType, {})
                
                if png_data:
                    # 获取字节数据
                    bytes_data = png_data.bytes()
                    length = png_data.length()
                    
                    # 转换为Python字节
                    raw_data = bytes_data[:length]
                    
                    # 保存到文件
                    with open(filepath, 'wb') as f:
                        f.write(raw_data)
                    print(f"截图已保存到: {filepath}")
                    
                    base64_encoded = base64.b64encode(raw_data).decode('utf-8')
                    
                    print(f"截图成功 (NSBitmapImageRep方法)，数据大小: {len(base64_encoded)} 字符")
                    return f"data:image/png;base64,{base64_encoded}"
        except Exception as ns_error:
            print(f"NSBitmapImageRep处理失败: {str(ns_error)}")
        
        print("所有截图方法都失败了")
        return None
    
    except Exception as e:
        print(f"截图过程中出错: {str(e)}")
        return None

def perform_ocr_on_image(image_path):
    """
    使用PaddleOCR对图像执行OCR识别，返回识别的文本及其位置信息
    """
    try:
        # 初始化PaddleOCR（使用中英文模型）
        # 根据不同版本的PaddleOCR调整参数
        try:
            # 尝试新版API
            ocr = PaddleOCR(use_angle_cls=True, lang='ch')
            # 直接调用ocr方法，不指定cls参数
            result = ocr.ocr(image_path)
        except TypeError as e:
            # 如果失败，尝试旧版API
            print(f"尝试使用兼容模式初始化PaddleOCR: {e}")
            ocr = PaddleOCR(lang='ch')
            result = ocr.ocr(image_path)
        
        # 整理识别结果
        results = []
        
        print(f"PaddleOCR返回结果类型: {type(result)}")
        
        # 处理不同格式的返回结果
        if isinstance(result, list):
            # 处理列表格式的结果
            for idx, item in enumerate(result):
                if item is None:
                    continue
                    
                # 格式1: 列表包含字典（如paddlex格式）
                if isinstance(item, dict):
                    print(f"检测到字典格式结果，包含键: {item.keys()}")
                    
                    # 检查是否有'rec_texts'和'rec_scores'键
                    if 'rec_texts' in item and 'rec_scores' in item and 'rec_boxes' in item:
                        texts = item['rec_texts']
                        scores = item['rec_scores']
                        boxes = item['rec_boxes']
                        
                        # 确保三个列表长度相同
                        if len(texts) == len(scores) and len(texts) == len(boxes):
                            for i in range(len(texts)):
                                text = texts[i]
                                confidence = scores[i]
                                box = boxes[i]
                                
                                if confidence > 0.6 and text.strip():
                                    try:
                                        # 处理box坐标
                                        if hasattr(box, 'tolist'):  # 如果是numpy数组
                                            box_list = box.tolist()
                                        else:
                                            box_list = list(box)
                                        
                                        # 计算坐标
                                        if len(box_list) >= 4:
                                            # 假设box_list是[x1, y1, x2, y2]或类似格式
                                            if isinstance(box_list[0], (int, float)):
                                                # 一维数组格式 [x1, y1, x2, y2]
                                                x_coords = [box_list[0], box_list[2]]
                                                y_coords = [box_list[1], box_list[3]]
                                            else:
                                                # 二维数组格式 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                                                x_coords = [point[0] for point in box_list if isinstance(point, (list, tuple)) and len(point) >= 2]
                                                y_coords = [point[1] for point in box_list if isinstance(point, (list, tuple)) and len(point) >= 2]
                                            
                                            if x_coords and y_coords:
                                                left = min(x_coords)
                                                top = min(y_coords)
                                                width = max(x_coords) - left
                                                height = max(y_coords) - top
                                                x_center = left + width / 2
                                                y_center = top + height / 2
                                                
                                                results.append({
                                                    'text': text.strip(),
                                                    'confidence': confidence,
                                                    'x': x_center,
                                                    'y': y_center,
                                                    'left': left,
                                                    'top': top,
                                                    'width': width,
                                                    'height': height,
                                                    'box': box_list
                                                })
                                    except Exception as detail_error:
                                        print(f"处理字典格式文本区域时出错: {str(detail_error)}")
                    
                # 格式2: 标准PaddleOCR格式 [[文本框, (文本, 置信度)], ...]
                elif isinstance(item, list):
                    for word_info in item:
                        if isinstance(word_info, list) and len(word_info) >= 2:
                            try:
                                box = word_info[0]  # 文本框坐标
                                if isinstance(word_info[1], tuple) and len(word_info[1]) >= 2:
                                    text = word_info[1][0]  # 识别的文本
                                    confidence = word_info[1][1]  # 置信度
                                    
                                    if confidence > 0.6 and text.strip():
                                        # 计算文本区域的中心点坐标
                                        x_coords = [point[0] for point in box if isinstance(point, (list, tuple)) and len(point) >= 2]
                                        y_coords = [point[1] for point in box if isinstance(point, (list, tuple)) and len(point) >= 2]
                                        
                                        if x_coords and y_coords:
                                            left = min(x_coords)
                                            top = min(y_coords)
                                            width = max(x_coords) - left
                                            height = max(y_coords) - top
                                            x_center = left + width / 2
                                            y_center = top + height / 2
                                            
                                            results.append({
                                                'text': text.strip(),
                                                'confidence': confidence,
                                                'x': x_center,
                                                'y': y_center,
                                                'left': left,
                                                'top': top,
                                                'width': width,
                                                'height': height,
                                                'box': box
                                            })
                            except Exception as detail_error:
                                print(f"处理列表格式文本区域时出错: {str(detail_error)}")
        
        # 打印识别到的文本
        for i, res in enumerate(results):
            print(f"识别文本 {i+1}: '{res['text']}' (置信度: {res['confidence']:.2f}) 在位置 ({res['x']:.1f}, {res['y']:.1f})")
        
        print(f"最终处理得到 {len(results)} 个有效文本区域")
        return results
    except Exception as e:
        print(f"PaddleOCR处理失败: {str(e)}")
        # 输出详细的错误信息以帮助调试
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return []

def find_text_position_in_window(window_info, target_text, timeout=10):
    """
    在指定窗口中查找目标文本的位置
    """
    start_time = time.time()
    
    # 打印窗口信息以调试
    print(f"窗口信息: 位置=({window_info['left']}, {window_info['top']}), 大小=({window_info['width']}, {window_info['height']})")
    print(f"查找目标文本: '{target_text}'")
    
    # 针对macOS的坐标系校准参数
    # 微信窗口通常有标题栏和边框，需要进行补偿
    border_offset_x = 0  # 水平边框补偿
    border_offset_y = 0  # 垂直边框补偿
    
    # 获取应用名称进行特定应用的校准
    app_name = window_info.get('app_name', '').lower()
    
    # 针对微信等应用的特殊校准
    if '微信' in app_name or 'wechat' in app_name:
        # 在macOS上，微信窗口的Quartz坐标和实际可点击区域可能有差异
        # 添加边框补偿以提高准确性
        border_offset_x = 8  # 左侧边框补偿
        border_offset_y = 22  # 顶部标题栏补偿
        print(f"应用微信特定校准: 边框补偿 ({border_offset_x}, {border_offset_y})")
    
    while time.time() - start_time < timeout:
        # 捕获窗口截图
        screenshot_data = capture_window_screenshot(window_info)
        
        if not screenshot_data:
            time.sleep(1)
            continue
        
        # 获取保存的截图路径
        images_dir = os.path.join(os.path.dirname(__file__), 'images')
        app_name = window_info.get('app_name', 'unknown').replace(' ', '_')
        window_title = window_info.get('title', 'untitled').replace(' ', '_')
        timestamp = os.path.splitext(os.path.basename(__file__))[0]
        filepath = os.path.join(images_dir, f"{app_name}_{window_title}_{timestamp}.png")
        
        if os.path.exists(filepath):
            # 执行PaddleOCR识别
            ocr_results = perform_ocr_on_image(filepath)
            
            print(f"OCR识别到 {len(ocr_results)} 个文本区域")
            for result in ocr_results:
                print(f"  识别文本: '{result['text']}' (置信度: {result['confidence']:.2f}), OCR坐标: ({result['x']:.1f}, {result['y']:.1f})")
            
            # 查找目标文本，使用更宽松的匹配方式
            for result in ocr_results:
                # 直接匹配或子串匹配
                if target_text == result['text'] or target_text in result['text'] or result['text'] in target_text:
                    # 将OCR坐标系转换为屏幕坐标系，调整为使用相对坐标
                    # 对于微信小程序界面，需要特殊处理y轴坐标
                    ocr_x = float(result['x'])
                    ocr_y = float(result['y'])
                    
                    # 针对微信小程序界面的特殊校准
                    # x轴应用正常的边框补偿
                    screen_x = ocr_x + border_offset_x
                    
                    # y轴进行特殊调整，从OCR的大值(1000+)缩放到700+范围
                    # 通过减去一个固定偏移值来调整y坐标
                    y_offset_adjustment = 350  # 调整y轴偏移的固定值
                    screen_y = ocr_y - y_offset_adjustment + border_offset_y
                    
                    print(f"直接匹配成功: 目标'{target_text}' 匹配到 '{result['text']}'")
                    print(f"  OCR相对坐标: ({ocr_x:.1f}, {ocr_y:.1f})")
                    print(f"  应用边框补偿: ({border_offset_x}, {border_offset_y})")
                    print(f"  转换后屏幕坐标: ({screen_x:.1f}, {screen_y:.1f})")
                    return (screen_x, screen_y)
            
            # 如果直接匹配失败，尝试分词匹配
            import re
            target_words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', target_text)
            for target_word in target_words:
                if len(target_word) >= 2:  # 只匹配长度>=2的词
                    for result in ocr_results:
                        if target_word in result['text'] or result['text'] in target_word:
                            # 将OCR坐标系转换为屏幕坐标系，调整为使用相对坐标
                            # 对于微信小程序界面，需要特殊处理y轴坐标
                            ocr_x = float(result['x'])
                            ocr_y = float(result['y'])
                            
                            # 针对微信小程序界面的特殊校准
                            # x轴应用正常的边框补偿
                            screen_x = ocr_x + border_offset_x
                            
                            # y轴进行特殊调整，从OCR的大值(1000+)缩放到700+范围
                            # 通过减去一个固定偏移值来调整y坐标
                            y_offset_adjustment = 350  # 调整y轴偏移的固定值
                            screen_y = ocr_y - y_offset_adjustment + border_offset_y
                            
                            print(f"分词匹配成功: 目标'{target_word}' 匹配到 '{result['text']}'")
                            print(f"  OCR相对坐标: ({ocr_x:.1f}, {ocr_y:.1f})")
                            print(f"  应用边框补偿: ({border_offset_x}, {border_offset_y})")
                            print(f"  转换后屏幕坐标: ({screen_x:.1f}, {screen_y:.1f})")
                            return (screen_x, screen_y)
        
        time.sleep(1)
    
    print(f"在{timeout}秒内未能找到目标文本 '{target_text}'")
    return None

def click_at_position(x, y, click_type='left', duration=0.25):
    """
    控制鼠标点击指定位置
    
    参数:
    - x, y: 点击位置的屏幕坐标
    - click_type: 点击类型，'left'（左键）或'right'（右键）
    - duration: 移动鼠标的持续时间，使移动更平滑
    """
    try:
        print(f"移动鼠标到 ({x}, {y}) 并执行{click_type}点击")
        
        # 移动鼠标到指定位置
        pyautogui.moveTo(x, y, duration=duration)
        
        # 执行点击操作
        if click_type == 'left':
            pyautogui.click()
        elif click_type == 'right':
            pyautogui.rightClick()
        
        return True
    except Exception as e:
        print(f"鼠标点击操作失败: {str(e)}")
        return False

def call_multimodal_model(image_base64, prompt="仅输出这题答案"):
    """
    使用OpenAI客户端调用多模态大模型分析图片
    """
    try:
        completion = client.chat.completions.create(
            model=OPENAI_CONFIG['multimodal_model'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_base64}}
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        # 返回更详细的错误信息以便调试
        try:
            error_details = e.response.json() if hasattr(e, 'response') else "无详细错误信息"
            return f"调用模型失败: {str(e)}，详细错误: {error_details}"
        except:
            return f"调用模型失败: {str(e)}"

def get_wechat_windows(windows):
    """
    从窗口列表中筛选出微信窗口
    """
    wechat_windows = []
    print("开始筛选微信窗口...")
    
    for i, window in enumerate(windows):
        # 打印窗口字典的所有键，以便调试
        print(f"窗口 {i+1} 的字段: {window.keys()}")
        
        # 使用更安全的方式获取应用名称
        app_name = window.get('app_name', 'Unknown')
        
        # 根据打印的键确定正确的标题字段名
        title_keys = ['title', 'window_title', 'kCGWindowName']
        window_title = 'Unknown'
        for key in title_keys:
            if key in window:
                window_title = window[key]
                break
        
        print(f"检查窗口 {i+1}: 应用名称='{app_name}', 窗口标题='{window_title}'")
        
        # 使用更宽松的匹配条件来识别微信窗口
        if '微信' in app_name or 'WeChat' in app_name or 'Wechat' in app_name:
            wechat_windows.append(window)
            print(f"✓ 匹配到微信窗口: {app_name} - {window_title}")
    
    print(f"筛选完成，共找到 {len(wechat_windows)} 个微信窗口")
    return wechat_windows

def click_model_output(window_info, model_result):
    """
    点击模型输出的文本
    
    参数:
    - window_info: 窗口信息字典
    - model_result: 模型输出的文本结果
    """
    try:
        print(f"\n=== 开始点击模型输出文本 ===")
        print(f"模型输出结果: {model_result}")
        
        # 清理模型结果文本，去除特殊字符
        import re
        clean_result = re.sub(r'[\n\r\t\s]+', ' ', model_result).strip()
        
        # 尝试多种匹配策略
        # 策略1: 直接匹配整个结果（如果结果不太短）
        if len(clean_result) > 5 and len(clean_result) < 50:
            print(f"策略1: 尝试直接匹配结果")
            position = find_text_position_in_window(window_info, clean_result)
            if position:
                success = click_at_position(position[0], position[1])
                if success:
                    print(f"成功点击完整结果")
                    return True
        
        # 策略2: 提取数字和字母组合（通常是选项或答案）
        code_patterns = re.findall(r'[A-Za-z0-9]+', clean_result)
        for code in code_patterns:
            if len(code) >= 2:
                print(f"策略2: 尝试匹配代码/选项: '{code}'")
                position = find_text_position_in_window(window_info, code, timeout=5)
                if position:
                    success = click_at_position(position[0], position[1])
                    if success:
                        print(f"成功点击代码/选项 '{code}'")
                        return True
        
        # 策略3: 提取中文关键词
        chinese_keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', clean_result)
        for keyword in chinese_keywords[:3]:  # 只尝试前3个中文关键词
            print(f"策略3: 尝试匹配中文关键词: '{keyword}'")
            position = find_text_position_in_window(window_info, keyword, timeout=5)
            if position:
                success = click_at_position(position[0], position[1])
                if success:
                    print(f"成功点击中文关键词 '{keyword}'")
                    return True
        
        # 策略4: 分割成短句进行匹配
        sentences = re.split(r'[，。；！？]', clean_result)
        for sentence in sentences:
            if len(sentence.strip()) >= 4:
                print(f"策略4: 尝试匹配短句: '{sentence.strip()}'")
                position = find_text_position_in_window(window_info, sentence.strip(), timeout=5)
                if position:
                    success = click_at_position(position[0], position[1])
                    if success:
                        print(f"成功点击短句 '{sentence.strip()}'")
                        return True
        
        print("所有匹配策略均失败，未能找到并点击模型输出的文本")
        return False
    except Exception as e:
        print(f"点击模型输出过程中出错: {str(e)}")
        return False

if __name__ == "__main__":
    print("活跃应用窗口信息：")
    print("-" * 80)
    
    windows = get_window_info()
    
    if not windows:
        print("未检测到活跃的应用窗口")
    else:
        # 显示所有窗口信息
        for i, win in enumerate(windows, 1):
            print(f"[{i}] 应用：{win['app_name']}")
            print(f"   窗口标题：{win['title']}")
            print(f"   位置：({win['left']}, {win['top']})")
            print(f"   大小：{win['width']}x{win['height']}")
            print("-" * 80)
        
        # 查找微信窗口
        wechat_windows = get_wechat_windows(windows)
        if wechat_windows:
            print(f"\n检测到 {len(wechat_windows)} 个微信窗口")
            
            # 对每个微信窗口进行截图和分析
            for i, win in enumerate(wechat_windows, 1):
                print(f"\n正在处理微信窗口 {i}...")
                
                # 捕获截图
                print(f"正在捕获窗口截图...")
                screenshot = capture_window_screenshot(win)
                
                if screenshot:
                    print(f"截图成功，正在调用多模态模型分析...")
                    
                    # 调用多模态模型分析图片
                    result = call_multimodal_model(screenshot)
                    
                    print(f"\n=== 微信窗口 {i} 分析结果 ===")
                    print(result)
                    print("=" * 80)
                    
                    # 尝试点击模型输出的文本
                    click_model_output(win, result)
                else:
                    print(f"截图失败")
        else:
            print("未检测到微信窗口")
    
    print("\n程序执行完毕。要执行鼠标点击操作，需要确保：")
    print("1. 已安装必要的库：pip install pyautogui paddlepaddle paddleocr pillow")
    print("2. PaddleOCR已正确安装并配置")
    print("3. 目标窗口在屏幕上可见且未被遮挡")