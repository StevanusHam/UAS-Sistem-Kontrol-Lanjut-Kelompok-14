import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.exposure import histogram

# --- 1. Fungsi Otsu Thresholding (untuk referensi, menghitung between-class variance) ---
def otsu_threshold_between_variance(image_gray):
    """
    Menghitung nilai threshold Otsu dan between-class variance.
    """
    hist, bins = histogram(image_gray)
    total_pixels = image_gray.size

    best_threshold = 0
    max_between_variance = 0.0

    for t in range(256):  # Iterate through all possible threshold values (0-255)
        # Class 1: Pixels <= t
        w0 = np.sum(hist[:t]) / total_pixels
        mu0 = np.sum(bins[:t] * hist[:t]) / np.sum(hist[:t]) if np.sum(hist[:t]) > 0 else 0

        # Class 2: Pixels > t
        w1 = np.sum(hist[t:]) / total_pixels
        mu1 = np.sum(bins[t:] * hist[t:]) / np.sum(hist[t:]) if np.sum(hist[t:]) > 0 else 0

        # Global mean (average intensity of the whole image)
        mu_t = np.sum(bins * hist) / total_pixels

        # Calculate between-class variance
        between_variance = w0 * (mu0 - mu_t)**2 + w1 * (mu1 - mu_t)**2

        if between_variance > max_between_variance:
            max_between_variance = between_variance
            best_threshold = t

    return best_threshold, max_between_variance

# --- 2. Genetic Algorithm untuk Optimasi Threshold (menggunakan Between-Class Variance) ---
def genetic_algorithm_threshold_between(image_gray, target_between_variance_ratio=0.95, # Target sebagai rasio dari maks Otsu
                                        population_size=50, generations=100,
                                        mutation_rate=0.05):
    """
    Mencari nilai threshold terbaik menggunakan Genetic Algorithm
    dengan memaksimalkan between-class variance.
    """
    hist, bins = histogram(image_gray)
    total_pixels = image_gray.size
    
    # Hitung mean global untuk efisiensi
    mu_t_global = np.sum(bins * hist) / total_pixels

    # Opsional: Hitung maksimum between-class variance Otsu sebagai target referensi
    _, max_otsu_between_variance = otsu_threshold_between_variance(image_gray)
    target_between_variance_abs = max_otsu_between_variance * target_between_variance_ratio
    print(f"Target Between-Class Variance (absolut): {target_between_variance_abs:.4f} ({(target_between_variance_ratio*100):.0f}% dari Otsu max)")


    # Inisialisasi populasi acak (nilai threshold 0-255)
    population = np.random.randint(0, 256, population_size)
    best_overall_threshold = 0
    best_overall_between_variance = 0.0
    found_target = False

    for generation in range(generations):
        fitness_scores = []
        between_variances = []

        for threshold in population:
            # Hitung between-class variance untuk setiap individu (threshold)
            w0 = np.sum(hist[:threshold]) / total_pixels
            mu0 = np.sum(bins[:threshold] * hist[:threshold]) / np.sum(hist[:threshold]) if np.sum(hist[:threshold]) > 0 else 0

            w1 = np.sum(hist[threshold:]) / total_pixels
            mu1 = np.sum(bins[threshold:] * hist[threshold:]) / np.sum(hist[threshold:]) if np.sum(hist[threshold:]) > 0 else 0

            if w0 == 0 or w1 == 0:
                between_variance = 0.0 # Beri penalti jika kelas kosong
            else:
                between_variance = w0 * (mu0 - mu_t_global)**2 + w1 * (mu1 - mu_t_global)**2

            # Kebugaran (fitness) = between_variance (maksimalkan)
            fitness = between_variance
            fitness_scores.append(fitness)
            between_variances.append(between_variance)

        fitness_scores = np.array(fitness_scores)
        between_variances = np.array(between_variances)

        # Update best overall (cari yang paling tinggi)
        current_best_idx = np.argmax(between_variances)
        if between_variances[current_best_idx] > best_overall_between_variance:
            best_overall_between_variance = between_variances[current_best_idx]
            best_overall_threshold = population[current_best_idx]

        # Cek apakah target sudah tercapai
        if best_overall_between_variance >= target_between_variance_abs:
            found_target = True
            print(f"Target Between-Class Variance {target_between_variance_abs:.4f} tercapai pada generasi {generation+1} dengan threshold {best_overall_threshold} dan variance {best_overall_between_variance:.4f}")
            break

        # Seleksi: Pilih individu dengan fitness terbaik
        # Jika semua fitness 0, atasi pembagian nol
        if np.sum(fitness_scores) == 0:
            probabilities = np.ones(population_size) / population_size
        else:
            probabilities = fitness_scores / np.sum(fitness_scores)
        
        selected_indices = np.random.choice(population_size, size=population_size, p=probabilities)
        selected_population = population[selected_indices]

        # Persilangan (Crossover)
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1] if i+1 < population_size else selected_population[0]

            # Crossover titik tunggal
            crossover_point = np.random.randint(0, 8) # Threshold 0-255 (8 bit)
            mask = (1 << crossover_point) - 1
            child1 = (parent1 & ~mask) | (parent2 & mask)
            child2 = (parent2 & ~mask) | (parent1 & mask)

            new_population.extend([child1, child2])
        population = np.array(new_population[:population_size])

        # Mutasi
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(0, 8)
                population[i] = population[i] ^ (1 << mutation_point) # Flip bit
                population[i] = np.clip(population[i], 0, 255) # Pastikan dalam rentang 0-255

    if not found_target:
        print(f"Target Between-Class Variance {target_between_variance_abs:.4f} tidak tercapai setelah {generations} generasi.")
        print(f"Threshold terbaik yang ditemukan: {best_overall_threshold} dengan between-class variance: {best_overall_between_variance:.4f}")

    return best_overall_threshold, best_overall_between_variance

# --- 3. Main Program ---
if __name__ == "__main__":
    # Contoh gambar (ganti dengan path gambar Anda)
    image_path = 'D:/Sistem Kontrol Lanjut/WhatsApp Image 2025-06-30 at 14.11.24_5c165a17.jpg' # Contoh gambar grayscale
    # Jika Anda memiliki gambar lokal, gunakan:
    # image_path = 'path/to/your/image.jpg'

    try:
        # Baca gambar
        image = io.imread(image_path)
    except FileNotFoundError:
        print(f"Error: File '{image_path}' tidak ditemukan. Pastikan path gambar sudah benar.")
        exit()
    except Exception as e:
        print(f"Error saat membaca gambar: {e}")
        exit()

    # Konversi ke grayscale jika belum
    if image.ndim == 3:
        image_gray = color.rgb2gray(image)
        image_gray = (image_gray * 255).astype(np.uint8) # Konversi ke 0-255
    else:
        image_gray = image

    # --- Tampilkan Gambar Grayscale ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Gambar Grayscale')
    plt.axis('off')

    # --- Tampilkan Histogram ---
    hist, bins_center = histogram(image_gray)
    plt.subplot(1, 3, 2)
    plt.plot(bins_center, hist)
    plt.title('Histogram Gambar')
    plt.xlabel('Intensitas Piksel')
    plt.ylabel('Frekuensi')

    # --- Jalankan Genetic Algorithm untuk mendapatkan threshold terbaik ---
    print("\n--- Memulai Genetic Algorithm (Between-Class Variance) ---")
    # Anda bisa menyesuaikan target_between_variance_ratio.
    # Nilai 0.95 berarti GA akan mencoba mencari threshold yang memiliki
    # between-class variance setidaknya 95% dari nilai Otsu maksimal.
    best_ga_threshold, ga_between_variance = genetic_algorithm_threshold_between(image_gray, target_between_variance_ratio=0.95)
    print(f"\nThreshold terbaik dari GA: {best_ga_threshold}")
    print(f"Between-class variance dari GA: {ga_between_variance:.4f}")

    # --- Aplikasikan Threshold Terbaik ---
    binary_image_ga = image_gray > best_ga_threshold # Sesuaikan dengan definisi foreground/background Anda

    plt.subplot(1, 3, 3)
    plt.imshow(binary_image_ga, cmap='gray')
    plt.title(f'Hasil Threshold Terbaik (GA): {best_ga_threshold}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # --- Contoh penerapan threshold Otsu standar (untuk perbandingan) ---
    print("\n--- Otsu Thresholding Standar (untuk perbandingan) ---")
    otsu_t, otsu_var = otsu_threshold_between_variance(image_gray)
    print(f"Threshold Otsu standar: {otsu_t}")
    print(f"Between-class variance Otsu standar: {otsu_var:.4f}")

    # Tampilkan hasil Otsu standar
    binary_image_otsu = image_gray > otsu_t
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_image_otsu, cmap='gray')
    plt.title(f'Hasil Threshold Otsu Standar: {otsu_t}')
    plt.axis('off')
    plt.show()