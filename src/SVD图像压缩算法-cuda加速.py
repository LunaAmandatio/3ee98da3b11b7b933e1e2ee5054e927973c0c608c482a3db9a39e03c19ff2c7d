import numpy as np
import torch
from scipy.linalg import svd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os


class SVDCompression():
    def __init__(self, path=None):
        if path:
            self.load_image(path)

    def load_image(self, path):
        """加载图像并计算SVD (PyTorch优化版)"""
        self.image = Image.open(path)
        # 用 float32：消费级 GPU 上 fp32 比 fp64 快 1~2 个数量级，
        # 且重建后 clip 到 0-255 再转 uint8，精度完全够用。
        A = np.array(self.image, dtype=np.float32)
        self.image_path = path

        print(f"正在计算SVD分解: {os.path.basename(path)}")
        print(f"图像尺寸: {A.shape}")

        H, W, C = A.shape
        # (H, W, 3) -> (3, H, W)，让 torch 把 3 当 batch 维
        reshaped = np.transpose(A, (2, 0, 1))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        tensor = torch.from_numpy(reshaped).to(self.device)

        print("正在批量计算RGB通道SVD分解...")
        with torch.no_grad():
            U, s, Vt = torch.linalg.svd(tensor, full_matrices=False)
            # 预乘 U·diag(s)，rebuild 时只剩一次大矩阵乘
            Us = U * s.unsqueeze(1)            # (3, H, k_max)

        # GPU 上保留：(3, H, k_max) / (3, k_max) / (3, k_max, W)
        self.Us_gpu = Us
        self.s_gpu  = s
        self.Vt_gpu = Vt

        self.shape1 = self.shape2 = self.shape3 = (H, W)
        self.max_k = min(H, W)

        # 兼容 update_image 中按通道访问 s1/s2/s3 计算能量比例
        s_np = s.cpu().numpy()
        self.s1, self.s2, self.s3 = s_np[0], s_np[1], s_np[2]

        print(f"SVD分解完成！最大奇异值数量: {self.max_k}")
        print(f"奇异值形状: {self.s1.shape}")
        return True

    @torch.no_grad()
    def _rebuild_gpu(self, k):
        """在 GPU 上批量重建三通道，返回 (H, W, 3) uint8 numpy 数组"""
        # (3, H, k) @ (3, k, W) -> (3, H, W)
        rebuilt = self.Us_gpu[:, :, :k] @ self.Vt_gpu[:, :k, :]
        rebuilt = rebuilt.clamp_(0, 255).to(torch.uint8)
        # (3, H, W) -> (H, W, 3) 再回 CPU
        return rebuilt.permute(1, 2, 0).contiguous().cpu().numpy()

    def rebuild_by_percent(self, percent):
        """根据百分比重建图像"""
        # 三通道 shape 相同，k1==k2==k3，统一计算
        k = max(1, min(self.max_k, round(self.max_k * percent)))
        return self._rebuild_gpu(k), k, k, k

    def rebuild_by_k(self, k):
        """根据保留的前k个奇异值重建图像"""
        k = max(1, min(self.max_k, int(k)))
        return self._rebuild_gpu(k)


class SVDImageViewer:
    """基于tkinter的SVD图像查看器"""

    def __init__(self):
        self.compressor = None
        self.use_percent_mode = True
        self.current_image = None
        self.is_processing = False

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("SVD 图像压缩 - 实时拖动演示")
        self.root.geometry("1200x800")

        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')

        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：图像显示区域
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 图像显示标签
        self.image_label = ttk.Label(left_frame, text="请选择图片", font=('Arial', 16))
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 右侧：控制面板
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        # 控制面板内容
        control_panel = ttk.LabelFrame(right_frame, text="控制面板", padding=10)
        control_panel.pack(fill=tk.BOTH, expand=True)

        # 文件选择区域
        file_frame = ttk.LabelFrame(control_panel, text="文件", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self.file_path_var = tk.StringVar()
        self.file_path_var.set("未选择文件")
        file_path_label = ttk.Label(file_frame, textvariable=self.file_path_var,
                                    wraplength=280, foreground="gray")
        file_path_label.pack(fill=tk.X, pady=(0, 5))

        self.select_button = ttk.Button(file_frame, text="选择图片", command=self.select_image)
        self.select_button.pack(fill=tk.X)

        # 模式显示
        self.mode_label = ttk.Label(control_panel, text="当前模式: 百分比模式",
                                    font=('Arial', 12, 'bold'))
        self.mode_label.pack(pady=(10, 10))

        # 百分比模式滑块
        self.percent_frame = ttk.Frame(control_panel)
        self.percent_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.percent_frame, text="保留奇异值百分比:").pack()
        self.percent_var = tk.DoubleVar(value=0.95)
        self.percent_slider = ttk.Scale(self.percent_frame, from_=0.01, to=1.0,
                                        variable=self.percent_var, orient=tk.HORIZONTAL,
                                        command=self.on_slider_change)
        self.percent_slider.pack(fill=tk.X, pady=5)

        self.percent_value_label = ttk.Label(self.percent_frame, text="95.0%")
        self.percent_value_label.pack()

        # 前K个模式滑块
        self.k_frame = ttk.Frame(control_panel)
        ttk.Label(self.k_frame, text="保留前K个奇异值:").pack()
        self.k_var = tk.IntVar(value=100)
        self.k_slider = ttk.Scale(self.k_frame, from_=1, to=100,
                                  variable=self.k_var, orient=tk.HORIZONTAL,
                                  command=self.on_slider_change)
        self.k_slider.pack(fill=tk.X, pady=5)

        self.k_value_label = ttk.Label(self.k_frame, text="100/100")
        self.k_value_label.pack()
        self.k_frame.pack_forget()  # 初始隐藏

        # 模式切换按钮
        self.switch_button = ttk.Button(control_panel, text="切换模式",
                                        command=self.toggle_mode)
        self.switch_button.pack(pady=10)

        # 信息显示
        info_frame = ttk.LabelFrame(control_panel, text="信息", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.info_text = tk.Text(info_frame, height=10, width=35, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(self.info_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.info_text.yview)

        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪 - 请选择图片", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 进度条
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')

        # 启动主循环
        self.root.mainloop()

    def select_image(self):
        """选择图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        """加载图像"""
        if self.is_processing:
            return

        self.is_processing = True
        self.progress.pack(side=tk.BOTTOM, fill=tk.X)
        self.progress.start(10)
        self.status_bar.config(text=f"正在加载并计算SVD: {os.path.basename(file_path)}...")
        self.root.update()

        # 在新线程中处理，避免界面卡顿
        def process():
            try:
                self.compressor = SVDCompression(file_path)

                # 更新UI
                self.root.after(0, self.on_image_loaded, file_path)
            except Exception as e:
                self.root.after(0, self.on_load_error, str(e))

        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()

    def on_image_loaded(self, file_path):
        """图像加载完成"""
        self.file_path_var.set(file_path)

        # 更新滑块范围
        self.k_slider.config(to=self.compressor.max_k)
        k_init = min(100, self.compressor.max_k)
        self.k_var.set(k_init)
        self.k_value_label.config(text=f"{k_init}/{self.compressor.max_k}")

        # 重置模式
        if self.use_percent_mode:
            self.percent_var.set(0.95)
        else:
            self.k_var.set(k_init)

        # 更新显示
        self.update_image()

        # 停止进度条
        self.progress.stop()
        self.progress.pack_forget()
        self.is_processing = False
        self.status_bar.config(text=f"已加载: {os.path.basename(file_path)}")

    def on_load_error(self, error_msg):
        """加载错误处理"""
        self.progress.stop()
        self.progress.pack_forget()
        self.is_processing = False
        self.status_bar.config(text=f"加载失败: {error_msg}")
        messagebox.showerror("错误", f"无法加载图片:\n{error_msg}")

    def on_slider_change(self, event=None):
        """滑块变化时的回调"""
        if self.compressor:
            self.update_image()

    def update_image(self):
        """更新图像显示"""
        if not self.compressor:
            return

        try:
            # 根据模式重建图像
            if self.use_percent_mode:
                percent = self.percent_var.get()
                self.percent_value_label.config(text=f"{percent * 100:.1f}%")
                rebuilt_img, k1, k2, k3 = self.compressor.rebuild_by_percent(percent)

                # 计算能量保留比例
                total_energy1 = np.sum(self.compressor.s1 ** 2)
                energy1 = np.sum(self.compressor.s1[:k1] ** 2)
                total_energy2 = np.sum(self.compressor.s2 ** 2)
                energy2 = np.sum(self.compressor.s2[:k2] ** 2)
                total_energy3 = np.sum(self.compressor.s3 ** 2)
                energy3 = np.sum(self.compressor.s3[:k3] ** 2)
                energy_ratio = (energy1 + energy2 + energy3) / (total_energy1 + total_energy2 + total_energy3)

                # 计算压缩比
                total_original = (self.compressor.shape1[0] * self.compressor.shape1[1] +
                                  self.compressor.shape2[0] * self.compressor.shape2[1] +
                                  self.compressor.shape3[0] * self.compressor.shape3[1]) * 3
                total_compressed = (k1 * (self.compressor.shape1[0] + self.compressor.shape1[1]) +
                                    k2 * (self.compressor.shape2[0] + self.compressor.shape2[1]) +
                                    k3 * (self.compressor.shape3[0] + self.compressor.shape3[1]))
                compression_ratio = total_compressed / total_original

                info = f"模式: 百分比模式\n"
                info += f"保留比例: {percent * 100:.1f}%\n"
                info += f"能量保留: {energy_ratio:.2%}\n"
                info += f"奇异值数量:\n"
                info += f"  R通道: {k1}\n"
                info += f"  G通道: {k2}\n"
                info += f"  B通道: {k3}\n"
                info += f"压缩比: {compression_ratio:.2%}\n"
                info += f"数据量: {total_compressed / 1024:.1f}KB / {total_original / 1024:.1f}KB\n"
                info += f"图像尺寸: {self.compressor.shape1[1]}x{self.compressor.shape1[0]}"

            else:
                k = self.k_var.get()
                self.k_value_label.config(text=f"{k}/{self.compressor.max_k}")
                rebuilt_img = self.compressor.rebuild_by_k(k)

                # 计算能量保留比例
                total_energy1 = np.sum(self.compressor.s1 ** 2)
                energy1 = np.sum(self.compressor.s1[:k] ** 2)
                total_energy2 = np.sum(self.compressor.s2 ** 2)
                energy2 = np.sum(self.compressor.s2[:k] ** 2)
                total_energy3 = np.sum(self.compressor.s3 ** 2)
                energy3 = np.sum(self.compressor.s3[:k] ** 2)
                energy_ratio = (energy1 + energy2 + energy3) / (total_energy1 + total_energy2 + total_energy3)

                # 计算压缩比
                total_original = (self.compressor.shape1[0] * self.compressor.shape1[1] +
                                  self.compressor.shape2[0] * self.compressor.shape2[1] +
                                  self.compressor.shape3[0] * self.compressor.shape3[1]) * 3
                total_compressed = (k * (self.compressor.shape1[0] + self.compressor.shape1[1]) +
                                    k * (self.compressor.shape2[0] + self.compressor.shape2[1]) +
                                    k * (self.compressor.shape3[0] + self.compressor.shape3[1]))
                compression_ratio = total_compressed / total_original

                info = f"模式: 前K个模式\n"
                info += f"保留奇异值: {k}/{self.compressor.max_k}\n"
                info += f"保留比例: {k / self.compressor.max_k * 100:.1f}%\n"
                info += f"能量保留: {energy_ratio:.2%}\n"
                info += f"压缩比: {compression_ratio:.2%}\n"
                info += f"数据量: {total_compressed / 1024:.1f}KB / {total_original / 1024:.1f}KB\n"
                info += f"图像尺寸: {self.compressor.shape1[1]}x{self.compressor.shape1[0]}"

            # 更新信息显示
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)

            # 转换图像为tkinter可显示格式
            img_pil = Image.fromarray(rebuilt_img)

            # 调整图像大小以适应窗口
            display_size = (800, 600)
            img_pil.thumbnail(display_size, Image.Resampling.LANCZOS)

            self.current_image = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=self.current_image)

            # 更新状态栏
            self.status_bar.config(text=f"更新完成 - {info.split(chr(10))[0]}")

        except Exception as e:
            self.status_bar.config(text=f"错误: {str(e)}")

    def toggle_mode(self):
        """切换模式"""
        self.use_percent_mode = not self.use_percent_mode

        if self.use_percent_mode:
            self.mode_label.config(text="当前模式: 百分比模式")
            self.percent_frame.pack(fill=tk.X, pady=5)
            self.k_frame.pack_forget()
        else:
            self.mode_label.config(text="当前模式: 前K个模式")
            self.k_frame.pack(fill=tk.X, pady=5)
            self.percent_frame.pack_forget()

        self.update_image()

if __name__ == '__main__':
    # 在主线程预热 cuBLAS：cuBLAS handle 是 per-thread 的，
    # SVD 跑在后台线程、rebuild 跑在主线程，必须在主线程显式触发一次
    # matmul，否则首帧会出现 "no current CUDA context" 警告。
    if torch.cuda.is_available():
        _warm = torch.zeros((2, 2), device='cuda')
        _ = (_warm @ _warm).cpu()
        del _warm
    # 启动界面（不预先加载图片）
    viewer = SVDImageViewer()