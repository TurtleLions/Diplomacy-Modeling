import os
from diplomacy import Game

def visualize_log(log_path, output_dir="./visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    game = Game()
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    current_power = None
    orders_by_power = {}
    phase_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # When we hit a new phase, process the collected orders
        if line.startswith("--- Phase"):
            if orders_by_power:
                for pwr, ords in orders_by_power.items():
                    game.set_orders(pwr, ords)
                
                # Render the map with the order arrows before adjudicating
                output_path = os.path.join(output_dir, f"step_{phase_count:03d}_{game.current_phase}.svg")
                
                svg_data = game.render()
                
                # Handle both string and byte return types just in case
                if isinstance(svg_data, bytes):
                    with open(output_path, 'wb') as svg_file:
                        svg_file.write(svg_data)
                else:
                    with open(output_path, 'w', encoding='utf-8') as svg_file:
                        svg_file.write(svg_data)
                        
                game.process()
                phase_count += 1
                orders_by_power = {}
                
        elif line.endswith("Orders"):
            current_power = line.split()[0]
            if current_power not in orders_by_power:
                orders_by_power[current_power] = []
                
        # Collect actual order strings
        elif current_power and line != "(No valid orders)" and not line.startswith("FINAL") and not line.startswith("==="):
            orders_by_power[current_power].append(line)
            
    # Process the final turn if any orders were left in the buffer
    if orders_by_power:
        for pwr, ords in orders_by_power.items():
            game.set_orders(pwr, ords)
            
        output_path = os.path.join(output_dir, f"step_{phase_count:03d}_{game.current_phase}.svg")
        svg_data = game.render()
        
        if isinstance(svg_data, bytes):
            with open(output_path, 'wb') as svg_file:
                svg_file.write(svg_data)
        else:
            with open(output_path, 'w', encoding='utf-8') as svg_file:
                svg_file.write(svg_data)
            
    print(f"Finished rendering {phase_count + 1} phases to {output_dir}")

if __name__ == "__main__":
    # Point this to whatever log file you want to watch
    visualize_log("./eval_games/eval_game_update_50.txt")