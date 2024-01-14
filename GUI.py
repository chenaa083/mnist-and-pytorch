# -*- coding:utf-8 -*-
import test_model
import PySimpleGUI as sg


# 定义 GUI 布局
layout = [
    [sg.Text("Hello, PySimpleGUI!")],
    [sg.Text("choose your picture"), sg.Input(), sg.FileBrowse()],
    [sg.Button("OK")]
]

# 创建窗口
window = sg.Window("My GUI", layout, font=("宋体", 15),default_element_size=(50, 1))

# 事件循环
while True:
    event, values = window.read()

    # 处理事件
    if event == "OK":
        selected_file = values[0]
        print(f'选择的文件是: {selected_file}')
        # 在这里添加运行 test_model.py 的代码
        try:
            test_model.predict(selected_file)  # 运行 test_model.py
        except Exception as e:
            sg.popup_error(f"Error running a.py: {e}")
    if event == sg.WINDOW_CLOSED:
        break
# 关闭窗口
window.close()
