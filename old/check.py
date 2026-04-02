import json
import multiprocessing as smp
import time
from diplomacy import Game
from diplomacy_helpers import build_global_vocab

# Global variable for background workers
worker_vocab = None

def init_worker(vocab):
    global worker_vocab
    worker_vocab = vocab

def _scan_game(line):
    global worker_vocab
    
    try:
        game_data = json.loads(line)
    except json.JSONDecodeError:
        return 0
        
    if game_data.get('map', 'standard') != 'standard':
        return 0
        
    game_engine = Game()
    provinces = list(game_engine.map.locs)
    powers = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
    
    max_indices_in_game = 0
    
    for phase in game_data.get('phases', []):
        game_engine.set_state(phase['state'])
        
        for power in powers:
            # We must replicate the exact logic of get_global_action_mask
            orderable_locs = game_engine.get_orderable_locations(power)
            valid_indices_count = 0
            
            for prov in provinces:
                prov_has_order = False
                if prov in orderable_locs:
                    for order in game_engine.get_all_possible_orders().get(prov, []):
                        if order in worker_vocab:
                            valid_indices_count += 1
                            prov_has_order = True
                
                # If no valid orders for this province, it defaults to 'NONE' (1 index)
                if not prov_has_order:
                    valid_indices_count += 1
                    
            if valid_indices_count > max_indices_in_game:
                max_indices_in_game = valid_indices_count
                
    return max_indices_in_game

def main():
    json_path = "./datasets/standard_no_press.jsonl"
    
    print("Loading Global Vocabulary...")
    order_to_idx, _ = build_global_vocab()
    
    num_cores = min(8, smp.cpu_count())
    print(f"Starting standard multiprocessing pool with {num_cores} workers...")
    
    absolute_max = 0
    games_processed = 0
    start_time = time.time()
    
    def line_generator():
        with open(json_path, 'r') as f:
            for line in f:
                if line.strip():
                    yield line

    with smp.Pool(processes=num_cores, initializer=init_worker, initargs=(order_to_idx,)) as pool:
        try:
            # chunksize=50 to make scanning millions of lines fast
            for max_in_game in pool.imap_unordered(_scan_game, line_generator(), chunksize=50):
                games_processed += 1
                
                if max_in_game > absolute_max:
                    absolute_max = max_in_game
                    print(f"--> [NEW RECORD] Found a turn with {absolute_max} valid moves! (Game {games_processed})")
                    
                if games_processed % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = games_processed / elapsed
                    print(f"Scanned {games_processed} games... (Current Max: {absolute_max}) [{rate:.0f} games/s]")
                    
        except KeyboardInterrupt:
            print("\nScan canceled early by user.")
            pool.terminate()
            pool.join()
            
    print("-" * 40)
    print(f"SCAN COMPLETE.")
    print(f"Total Games Scanned: {games_processed}")
    print(f"ABSOLUTE MAXIMUM VALID INDICES: {absolute_max}")
    print("-" * 40)

if __name__ == "__main__":
    main()