import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from font import parse_txt_dataset

class YSA_Arayuz:
    def __init__(self, pencere, ysa_modeli, etiketler, satir, sutun):
        self.pencere = pencere
        self.ysa = ysa_modeli
        self.labels = etiketler
        self.satir = satir
        self.sutun = sutun
        self.arayuz_kur()

    def arayuz_kur(self):
        self.sol_frame = tk.Frame(self.pencere, padx=20, pady=20)
        self.sol_frame.pack(side=tk.LEFT)
        
        self.matris_vars = []
        grid_frame = tk.Frame(self.sol_frame)
        grid_frame.pack(pady=10)

        for r in range(self.satir):
            satir_vars = []
            for c in range(self.sutun):
                var = tk.IntVar(value=0)
                btn_w = 4 if self.sutun <= 5 else 2
                btn_h = 2 if self.satir <= 7 else 1
                tk.Checkbutton(grid_frame, variable=var, indicatoron=False, 
                               width=btn_w, height=btn_h, selectcolor="black", bg="white").grid(row=r, column=c)
                satir_vars.append(var)
            self.matris_vars.append(satir_vars)

        self.orta_frame = tk.Frame(self.pencere, padx=20)
        self.orta_frame.pack(side=tk.LEFT)

        tk.Button(self.orta_frame, text="SİSTEMİ SIFIRLA", command=self.sistemi_sifirla, 
                  width=20, bg="red", fg="white", font=("Arial", 10, "bold")).pack(pady=10)
        self.lr_entry = tk.Entry(self.orta_frame, justify='center')
        self.lr_entry.insert(0, "0.01")
        self.lr_entry.pack(pady=5)



        tk.Label(self.orta_frame, text="Epoch (İterasyon) Sayısı:", font=("Arial", 10)).pack(pady=(10, 0))
        self.epoch_entry = tk.Entry(self.orta_frame, justify='center')
        self.epoch_entry.insert(0, "2000")
        self.epoch_entry.pack(pady=5)

        tk.Label(self.orta_frame, text="Gizli Nöron Sayısı:", font=("Arial", 10)).pack(pady=(10, 0))
        self.hidden_n_entry = tk.Entry(self.orta_frame, justify='center')
        self.hidden_n_entry.insert(0, "40")
        self.hidden_n_entry.pack(pady=5)

        tk.Button(self.orta_frame, text="Veri Seti Seç (.txt)", command=self.dosya_sec, 
                  width=20, bg="orange", font=("Arial", 10, "bold")).pack(pady=15)
        
        tk.Button(self.orta_frame, text="Ağı Eğit", command=self.egitimi_baslat, 
                  width=20, bg="lightblue").pack(pady=5)
        
        tk.Button(self.orta_frame, text="Tahmin Et", command=self.tahmin_yap, 
                  width=20, bg="lightgreen").pack(pady=5)
        
        tk.Button(self.orta_frame, text="Temizle", command=self.temizle, 
                  width=20).pack(pady=5)
        
        self.sonuc_label = tk.Label(self.orta_frame, text="Tahmin: ?", 
                                   font=("Arial", 18, "bold"), fg="red")
        self.sonuc_label.pack(pady=20)

        self.sag_frame = tk.Frame(self.pencere, padx=20)
        self.sag_frame.pack(side=tk.RIGHT)
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Hata Payı (MSE)")
        self.line, = self.ax.plot([], [], 'r-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.sag_frame)
        self.canvas.get_tk_widget().pack()

    def dosya_sec(self):
        dosya_yolu = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if dosya_yolu:
            try:
                beklenen = self.satir * self.sutun
                self.X_train, self.Y_train, self.labels = parse_txt_dataset(dosya_yolu, beklenen)
               
                yeni_gizli_n = int(self.hidden_n_entry.get())
                
                from yapaysiniragi import YapaySinirAgi
                self.ysa = YapaySinirAgi(
                    giris_n=beklenen, 
                    gizli_n=yeni_gizli_n, 
                    cikis_n=len(self.labels), 
                    ogrenme_hizi=float(self.lr_entry.get())
                )
                
                messagebox.showinfo("Başarılı", f"{len(self.X_train)} örnek yüklendi ve model güncellendi!")
            except Exception as e:
                messagebox.showerror("Hata", f"Dosya hatası: {e}")

    def egitimi_baslat(self):
        if self.X_train is None:
            messagebox.showwarning("Uyarı", "Önce bir veri seti seçmelisiniz!")
            return

        try:
            lr_degeri = float(self.lr_entry.get())
            epoch_degeri = int(self.epoch_entry.get())
            self.ysa.lr = lr_degeri
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli sayısal değerler giriniz!")
            return

        mse_listesi = []
        for epoch in range(epoch_degeri + 1):
            indices = np.arange(self.X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = self.X_train[indices]
            Y_shuffled = self.Y_train[indices]
        
            toplam_hata = 0
            for i in range(len(X_shuffled)):
                x = X_shuffled[i:i+1]
                y = Y_shuffled[i:i+1]
                hata = self.ysa.egit(x, y)
                toplam_hata += hata
        
            guncelleme_araligi = max(1, epoch_degeri // 40)
            if epoch % guncelleme_araligi == 0:
                ortalama_mse = toplam_hata / len(X_shuffled)
                mse_listesi.append(ortalama_mse)
                self.line.set_data(range(len(mse_listesi)), mse_listesi)
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw()
                self.pencere.update()
            
        messagebox.showinfo("Eğitim Tamamlandı!", f"{epoch_degeri} epoch sonunda model hazır.")

    def tahmin_yap(self):
        matris = np.array([[self.matris_vars[r][c].get() for c in range(self.sutun)] for r in range(self.satir)])

        coords = np.argwhere(matris == 1)
        
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            kirpilmis = matris[y_min:y_max+1, x_min:x_max+1]
            h, w = kirpilmis.shape
            
            normalize_vektor = np.zeros((self.satir, self.sutun))
            
            for r in range(self.satir):
                for c in range(self.sutun):
                    orig_r = int((r + 0.5) * h / self.satir)
                    orig_c = int((c + 0.5) * w / self.sutun)
                    orig_r = min(orig_r, h - 1)
                    orig_c = min(orig_c, w - 1)
                    normalize_vektor[r, c] = kirpilmis[orig_r, orig_c]
            
            girdi = normalize_vektor.flatten().reshape(1, self.satir * self.sutun).astype(np.float32)
        else:
            return 

        cikti = self.ysa.forward(girdi)
        tahmin_index = np.argmax(cikti)
        
        if self.labels and tahmin_index < len(self.labels):
            harf_sonuc = self.labels[tahmin_index]
            self.sonuc_label.config(text=f"Tahmin: {harf_sonuc}")
    def sistemi_sifirla(self):
        if messagebox.askyesno("Onay", "Tüm hafıza silinecek?"):
            try:
                yeni_gizli_n = int(self.hidden_n_entry.get())
                self.X_train = self.Y_train = None
                cikis_sayisi = len(self.labels) if self.labels else 26
                from yapaysiniragi import YapaySinirAgi
                self.ysa = YapaySinirAgi(
                    giris_n=self.satir * self.sutun, 
                    gizli_n=yeni_gizli_n, 
                    cikis_n=cikis_sayisi, 
                    ogrenme_hizi=float(self.lr_entry.get())
                )
                
                self.temizle()
                messagebox.showinfo("Başarılı", f"Model {yeni_gizli_n} gizli nöron ile sıfırlandı!")
            except ValueError:
                messagebox.showerror("Hata", "Gizli nöron sayısı tam sayı olmalıdır!")
            
            messagebox.showinfo("Sıfırlandı", "Uygulama tertemiz hale getirildi. Yeni veri seti yükleyebilirsiniz.")
    def temizle(self):
        for r in range(self.satir):
            for c in range(self.sutun): 
                self.matris_vars[r][c].set(0)
        self.sonuc_label.config(text="Tahmin: ?")
