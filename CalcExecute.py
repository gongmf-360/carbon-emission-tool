import time
import tkinter as tk
from tkinter import ttk, messagebox
import configparser
from MLP_linear_eval import eval_carbon_emission



def get_input():
    try:
        user_input = 0 if len(entry.get().strip()) == 0 else float(entry.get())
        user_input2 = 0 if len(entry2.get().strip()) == 0 else float(entry2.get())
        user_input3 = 0 if len(entry3.get().strip()) == 0 else float(entry3.get())
        user_input4 = 0 if len(entry4.get().strip()) == 0 else float(entry4.get())
    except Exception as e:
        print(f"发生了一个错误：{e}")
        messagebox.showwarning("警告", "请输入数字！")
    print("你输入的是:", user_input,  user_input2, user_input3,  user_input4)
    button.config(state=tk.DISABLED)
    # 显示计算中，等待结果输出
    output_entry_total.config(text="计算中...")
    output_entry_total_eval.config(text="计算中...")

    config = configparser.ConfigParser()
    # 使用 'r' 模式和 utf-8 编码打开文件
    with open('coefficient_config.ini', 'r', encoding='utf-8') as configfile:
        config.read_file(configfile)
    value = float(config.get('coefficient', 'power'))
    value2 = float(config.get('coefficient', 'gas'))
    value3 = float(config.get('coefficient', 'aluminum'))
    value4 = float(config.get('coefficient', 'steel'))


    #  更新进度条的值
    for i in range(101):
        progressbar['value'] = i
        root.update_idletasks()  # 更新 GUI
        root.update()            # 更新 GUI
        # 这里可以加入一些延时，以便观察进度条的变化
        time.sleep(0.02)
    button.config(state=tk.ACTIVE)
    output_total = user_input*value + user_input2*value2 + user_input3*value3 + user_input4*value4


    # 显示结果
    output_label_detail_1.config(text="电力碳排放量： " + str(user_input*value) + " CO₂ e")
    output_label_detail_2.config(text="天然气碳排放量： " + str(user_input2*value2) + " CO₂ e")
    output_label_detail_3.config(text="铝碳排放量： " + str(user_input3*value3) + " CO₂ e")
    output_label_detail_4.config(text="钢碳排放量： " + str(user_input4*value4) + " CO₂ e")


    output_entry_total.config(text=output_total)
    # 模型训练结果
    pre_result = eval_carbon_emission([user_input, user_input2, user_input3, user_input4])
    output_entry_total_eval.config(text=pre_result)


# 创建主窗口
root = tk.Tk()

# 设置窗口大小为 500x300 像素 , 显示在屏幕的 (500, 500) 坐标位置
root.geometry("500x300+500+500")

# 禁用窗口的大小调整功能
# root.resizable(False, False)

root.title('碳足迹计算工具')

label = tk.Label(root, text="电力:")
unit_label = tk.Label(root, text="kWh")
entry = tk.Entry(root, width=16)

label2 = tk.Label(root, text="天然气:")
unit_label2 = tk.Label(root, text="m³")
entry2 = tk.Entry(root, width=16)

label3 = tk.Label(root, text="铝:")
unit_label3 = tk.Label(root, text="kg")
entry3 = tk.Entry(root, width=16)

label4 = tk.Label(root, text="钢:")
unit_label4 = tk.Label(root, text="kg")
entry4 = tk.Entry(root, width=16)



label.grid(row=0, column=0)
unit_label.grid(row=0, column=2)
label2.grid(row=0, column=3)
label3.grid(row=1, column=0)
unit_label3.grid(row=1, column=2)
label4.grid(row=1, column=3)

entry.grid(row=0, column=1)
entry2.grid(row=0, column=4)
unit_label2.grid(row=0, column=5)
entry3.grid(row=1, column=1)
entry4.grid(row=1, column=4)
unit_label4.grid(row=1, column=5)


button = tk.Button(root, text="开始计算", command=get_input)
tk.Label(root, text="").grid(row=2, column=3)  # 空白行
button.grid(row=3, column=1)

# 创建进度条控件
progressbar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate")
# 跨越三列的标签
progressbar.grid(row=3, column=2, columnspan=3)
# 设置进度条的最大值
progressbar['maximum'] = 100

tk.Label(root, text="").grid(row=4, column=3)  # 空白行
output_label = tk.Label(root, text="各个能源和原材料的碳排放量")
output_label.grid(row=5, column=0, columnspan=2)

## 各能源和原料排放量
output_label_detail_1 = tk.Label(root, text="电力碳排放量：0 CO₂ e")
output_label_detail_1.grid(row=7, column=0, columnspan=2)

output_label_detail_2 = tk.Label(root, text="天然气碳排放量：0 CO₂ e")
output_label_detail_2.grid(row=7, column=2, columnspan=2)

output_label_detail_3 = tk.Label(root, text="铝碳排放量：0 CO₂ e")
output_label_detail_3.grid(row=8, column=0, columnspan=2)

output_label_detail_4 = tk.Label(root, text="钢碳排放量：0 CO₂ e")
output_label_detail_4.grid(row=8, column=2, columnspan=2)


##  总排放量
tk.Label(root, text="").grid(row=9, column=3)  # 空白行
output_label2 = tk.Label(root, text="总碳排放量（理论值）：")
output_label2.grid(row=10, column=0, columnspan=2)
output_entry_total = tk.Label(root, text="待计算")
output_entry_total.grid(row=10, column=2)
tk.Label(root, text="CO₂ e").grid(row=10, column=4, columnspan=1)  # 单位

output_label3 = tk.Label(root, text="总碳排放量（预估值）：")
output_label3.grid(row=12, column=0, columnspan=2)
output_entry_total_eval = tk.Label(root, text="待计算")
output_entry_total_eval.grid(row=12, column=2)
tk.Label(root, text="CO₂ e").grid(row=12, column=4, columnspan=1)  # 单位

root.mainloop()