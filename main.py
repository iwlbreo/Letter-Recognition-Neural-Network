import tkinter as tk
from tkinter import simpledialog
from yapaysiniragi import YapaySinirAgi
from arayuz import YSA_Arayuz

def ana_program():
    root = tk.Tk()
    root.withdraw() 


    satir = simpledialog.askinteger("Matris Yapısı", "Satır sayısı", initialvalue=7)
    sutun = simpledialog.askinteger("Matris Yapısı", "Sütun sayısı", initialvalue=5)
    
    if not satir or not sutun: return

    giris_n = satir * sutun
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") 
    ysa_modeli = YapaySinirAgi(giris_n=giris_n, gizli_n=40, cikis_n=len(labels), ogrenme_hizi=0.01)
    
    root.deiconify()
    root.title(f"YSA Harf Tanıma ({satir}x{sutun})")
    app = YSA_Arayuz(root, ysa_modeli, labels, satir, sutun)
    
    root.mainloop()

if __name__ == "__main__":
    ana_program()
