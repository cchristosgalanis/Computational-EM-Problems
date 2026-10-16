import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Φόρτωση των δεδομένων από το εξαγόμενο CSV
df = pd.read_csv('FDTD_heatmap.csv')

# Μετατροπή της 1D λίστας σε 2D πίνακα για το heatmap
# y-axis (index): t, x-axis (columns): x, color (values): E
heatmap_data = df.pivot(index='t', columns='x', values='E')

# Δημιουργία γραφήματος
plt.figure(figsize=(10, 6))

# Χρήση colormap 'jet' (ή 'turbo') για να ταιριάζει με την αισθητική των PINNs papers
ax = sns.heatmap(heatmap_data, 
                 cmap='jet', 
                 cbar_kws={'label': 'Electric Field Amplitude E(x,t)'})

# Αντιστροφή άξονα y ώστε το t=0 να είναι στο κάτω μέρος
ax.invert_yaxis()

# Καθαρισμός και προσαρμογή των ετικετών στους άξονες (ticks)
num_x_ticks = 6
num_y_ticks = 5

# Ορισμός θέσεων και τιμών για τον άξονα x (Space: 0.0 έως 5.0)
x_tick_positions = np.linspace(0, len(heatmap_data.columns) - 1, num_x_ticks)
x_tick_labels = np.round(np.linspace(0, 5.0, num_x_ticks), 1)
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_tick_labels, rotation=0)

# Ορισμός θέσεων και τιμών για τον άξονα y (Time: 0.0 έως 2.0)
y_tick_positions = np.linspace(0, len(heatmap_data.index) - 1, num_y_ticks)
y_tick_labels = np.round(np.linspace(0, 2.0, num_y_ticks), 1)
ax.set_yticks(y_tick_positions)
ax.set_yticklabels(y_tick_labels, rotation=0)

# Τίτλοι και ετικέτες
plt.title('Real Solution (FDTD Method)', fontsize=14, pad=15)
plt.xlabel('Space (x)', fontsize=12)
plt.ylabel('Time (t)', fontsize=12)

# Εξαγωγή και προβολή
plt.tight_layout()
plt.savefig('FDTD_Real_Solution_Heatmap.png', dpi=300, bbox_inches='tight')
plt.show()