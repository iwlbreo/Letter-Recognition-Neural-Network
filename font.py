import numpy as np

def parse_txt_dataset(file_path, beklenen_boyut):
    
    X = []
    Y_labels = []
    unique_labels = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line or ';' not in line or line.startswith('#'):
                continue
                
            bit_str, label = line.split(';')
            
            try:
                vektor = [int(b) for b in bit_str]
            except ValueError:
                continue
            
            if len(vektor) == beklenen_boyut:
                X.append(vektor)
                Y_labels.append(label)
                if label not in unique_labels:
                    unique_labels.append(label)
            else:
                print(f"Uyarı: {label} harfi hatalı boyutta ({len(vektor)}), beklenen: {beklenen_boyut}. Atlandı.")

        if not X:
            print("Hata: Hiç geçerli veri bulunamadı!")
            return None, None, []

        X_array = np.array(X, dtype=np.float32)
        unique_labels.sort()
        
        Y_one_hot = np.zeros((len(Y_labels), len(unique_labels)), dtype=np.float32)
        for i, label in enumerate(Y_labels):
            idx = unique_labels.index(label)
            Y_one_hot[i, idx] = 1
            
        return X_array, Y_one_hot, unique_labels

    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None, None, []