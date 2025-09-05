import matplotlib.pyplot as plt
import numpy as np

# ===================================================================
# 1. Plot Styling and LaTeX Configuration
# ===================================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"], # Or "Computer Modern Roman"
    "axes.titlepad": 20, 
    "figure.constrained_layout.use": True
})

# ===================================================================
# 2. UAV Data Definition & Styling
# ===================================================================

uav_data = {
    "Morph": {"payload": 0.5, "cruisePower": 90.4, "vtolPower": 52.56},
    "Fixed": {"payload": 0.5, "cruisePower": 313.888, "vtolPower": 153.845},
    "E-Hang 184": {"payload": 100, "cruisePower": 34600, "vtolPower": 42100},
    "Kitty Hawk Cora": {"payload": 181, "cruisePower": 63000, "vtolPower": 228000},
    "Lilium Jet": {"payload": 180, "cruisePower": 28000, "vtolPower": 187000}
}

# --- Define styles for each UAV ---
# FIXED: Keys are now plain text to match uav_data
styles = {
    'Morph': {'color': '#CC0000', 'linestyle': '-', 'label_pos': 'vtol', 'offset': (0.05, -0.75)},
    'Fixed': {'color': '#0066CC', 'linestyle': '-', 'label_pos': 'cruise', 'offset': (0, -0.2), 'va': 'top'},
    'E-Hang 184': {'color': '#008000', 'linestyle': '-', 'label_pos': 'vtol', 'offset': (0.05, 0.1), 'ha': 'left'},
    'Lilium Jet': {'color': '#660066', 'linestyle': '-', 'label_pos': 'cruise', 'offset': (-0.05, 0), 'ha': 'right'},
    'Kitty Hawk Cora': {'color': '#FF8000', 'linestyle': '-', 'label_pos': 'cruise', 'offset': (-0.1, 0.2), 'va': 'bottom'}
}


# ===================================================================
# 3. Create the Plot
# ===================================================================

fig, ax = plt.subplots(figsize=(8, 5))

for name, data in uav_data.items():
    cruise_eff = 1000 * data["payload"] / data["cruisePower"]
    vtol_eff = 1000 * data["payload"] / data["vtolPower"]
    
    x_points = [0, 1]
    y_points = [cruise_eff, vtol_eff]
    
    style = styles[name] # This lookup will now work correctly
    
    ax.plot(x_points, y_points, 
            color=style['color'], 
            linestyle=style['linestyle'], 
            linewidth=2.5, 
            marker='o', 
            markersize=8, 
            markerfacecolor='white', 
            markeredgewidth=2.5,
            clip_on=False)
            
    if style['label_pos'] == 'cruise':
        label_x, label_y = 0, cruise_eff
    else:
        label_x, label_y = 1, vtol_eff

    label_text = name
        
    ax.text(label_x + style.get('offset', (0,0))[0], 
            label_y + style.get('offset', (0,0))[1],
            label_text,
            color=style['color'],
            fontsize=16,
            ha=style.get('ha', 'center'),
            va=style.get('va', 'center'))

# ===================================================================
# 4. Final Formatting
# ===================================================================

ax.set_title(r'\textbf{Hybrid Flight Mission Efficiency}', fontsize=22)
ax.set_ylabel(r'\textbf{Payload Mass/Electrical Power (kg/kW)}',fontsize=18)
ax.set_xlabel(r'\textbf{Mission Profile}', fontsize=18)

ax.set_xticks([0, 1]) 
ax.set_xticklabels(['Cruise', 'VTOL'], fontsize=16) 
ax.tick_params(axis='x', length=0) 

ax.set_xlim(-0.3, 1.3)
ax.set_ylim(0, 14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("mission_efficiency_plot.pdf")
plt.show()