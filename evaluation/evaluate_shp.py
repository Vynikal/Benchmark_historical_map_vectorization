import json
from shapely.geometry import shape, LineString, MultiLineString, Point
from shapely.ops import unary_union
import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree
import os
from tqdm import tqdm
import sys
from datetime import datetime

class TeeOutput:
    """Class to write output to both console and file"""
    def __init__(self, *files):
        self.files = files

    def write(self, txt):
        for f in self.files:
            f.write(txt)
            f.flush()  # Ensure immediate writing

    def flush(self):
        for f in self.files:
            f.flush()

def read_shapefile_alternative(filepath):
    """Alternative way to read shapefiles using fiona directly"""
    import fiona
    
    geometries = []
    with fiona.open(filepath) as src:
        print(f"Reading {len(src)} features from {os.path.basename(filepath)}...")
        for feature in tqdm(src, desc="Loading geometries"):
            geom = shape(feature['geometry'])
            geometries.append(geom)
    return geometries

def flatten_lines(geoms):
    out = []
    print("Flattening geometries to LineStrings...")
    for g in tqdm(geoms, desc="Processing geometries"):
        if g is None or g.is_empty:
            continue
        if g.geom_type == "LineString":
            out.append(g)
        elif g.geom_type == "MultiLineString":
            out.extend(list(g.geoms))  # Use .geoms for newer shapely versions
        else:
            try:
                for part in g:
                    if part.geom_type == "LineString":
                        out.append(part)
            except Exception:
                pass
    return out

def total_length(lines):
    print("Calculating total length...")
    return sum(l.length for l in tqdm(lines, desc="Computing lengths"))

def matched_length(target_lines, matcher_union, tol):
    if matcher_union.is_empty:
        return 0.0
    
    print(f"Creating buffer with tolerance {tol}...")
    buf = matcher_union.buffer(tol, cap_style=1, join_style=1)
    
    matched = 0.0
    print("Computing matched lengths...")
    for l in tqdm(target_lines, desc="Processing intersections"):
        inter = l.intersection(buf)
        if inter.is_empty:
            continue
        if inter.geom_type == "LineString":
            matched += inter.length
        elif inter.geom_type == "MultiLineString":
            matched += sum(seg.length for seg in inter.geoms)
    return matched

def build_graph(lines):
    """Return graph, endpoints, degrees."""
    G = nx.Graph()
    print("Building network graph...")
    for ln in tqdm(lines, desc="Building graph"):
        coords = list(ln.coords)
        for i in range(len(coords) - 1):
            a, b = tuple(coords[i]), tuple(coords[i+1])
            G.add_node(a); G.add_node(b)
            G.add_edge(a, b)
    degs = dict(G.degree())
    endpoints = [pt for pt,d in degs.items() if d == 1]
    return G, endpoints, degs

def bad_intersections(lines):
    """Return list of intersection points where degree > 2."""
    print("Finding bad intersections...")
    G, _, degs = build_graph(lines)
    bad_pts = [Point(*coord) for coord, d in degs.items() if d > 2]
    return bad_pts

def bad_dangling(pred_endpoints, gt_endpoints, tol):
    """Return list of dangling endpoints in prediction not near GT endpoints."""
    if len(pred_endpoints) == 0:
        return []
    if len(gt_endpoints) == 0:
        return [Point(x,y) for x,y in pred_endpoints]
    
    print("Processing dangling endpoints...")
    
    # Convert endpoints to simple (x,y) coordinate lists
    def extract_coords(endpoints):
        coords = []
        for pt in tqdm(endpoints, desc="Extracting coordinates"):
            if isinstance(pt, tuple) and len(pt) >= 2:
                coords.append([float(pt[0]), float(pt[1])])
            elif hasattr(pt, '__iter__') and len(pt) >= 2:
                coords.append([float(pt[0]), float(pt[1])])
            else:
                print(f"Warning: Skipping invalid endpoint: {pt} (type: {type(pt)})")
        return coords
    
    pred_coords = extract_coords(pred_endpoints)
    gt_coords = extract_coords(gt_endpoints)
    
    if not pred_coords or not gt_coords:
        print("Warning: No valid coordinates found")
        return []
    
    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    pred_arr = np.array(pred_coords, dtype=float)
    gt_arr = np.array(gt_coords, dtype=float)
    
    print(f"pred_arr shape: {pred_arr.shape}")
    print(f"gt_arr shape: {gt_arr.shape}")
    
    # Final check for dimensions
    if pred_arr.shape[1] != 2 or gt_arr.shape[1] != 2:
        print(f"Error: Expected 2D coordinates, got pred: {pred_arr.shape}, gt: {gt_arr.shape}")
        return []
    
    try:
        print("Building KDTree...")
        tree = KDTree(gt_arr)
        print("Querying nearest neighbors...")
        dists, _ = tree.query(pred_arr, k=1)
        dists = dists.flatten()
        
        bad = []
        print("Identifying bad dangling points...")
        for i, dist in enumerate(tqdm(dists, desc="Checking distances")):
            if dist > tol:
                x, y = pred_coords[i]
                bad.append(Point(x, y))
        
        return bad
    except Exception as e:
        print(f"KDTree query failed: {e}")
        print(f"Trying manual distance calculation...")
        
        # Fallback: manual distance calculation
        bad = []
        for pred_coord in tqdm(pred_coords, desc="Manual distance calc"):
            min_dist = float('inf')
            for gt_coord in gt_coords:
                dist = np.sqrt((pred_coord[0] - gt_coord[0])**2 + (pred_coord[1] - gt_coord[1])**2)
                min_dist = min(min_dist, dist)
            
            if min_dist > tol:
                bad.append(Point(pred_coord[0], pred_coord[1]))
        
        return bad

def get_matched_segments(target_lines, matcher_union, tol):
    """Return the actual matched segments from target_lines that intersect with matcher_union"""
    if matcher_union.is_empty:
        return []
    
    print(f"Getting matched segments with tolerance {tol}...")
    buf = matcher_union.buffer(tol, cap_style=1, join_style=1)
    matched_segments = []
    
    for l in tqdm(target_lines, desc="Finding matches"):
        inter = l.intersection(buf)
        if inter.is_empty:
            continue
        
        if inter.geom_type == "LineString":
            matched_segments.append(inter)
        elif inter.geom_type == "MultiLineString":
            matched_segments.extend(list(inter.geoms))
        elif inter.geom_type == "GeometryCollection":
            # Handle GeometryCollection which may contain LineStrings
            for geom in inter.geoms:
                if geom.geom_type == "LineString":
                    matched_segments.append(geom)
    
    return matched_segments

def get_unmatched_segments(target_lines, matcher_union, tol):
    """Return the unmatched segments from target_lines that don't intersect with matcher_union"""
    if matcher_union.is_empty:
        return target_lines
    
    print(f"Getting unmatched segments with tolerance {tol}...")
    buf = matcher_union.buffer(tol, cap_style=1, join_style=1)
    unmatched_segments = []
    
    for l in tqdm(target_lines, desc="Finding unmatched"):
        # Get the part of the line that doesn't intersect with the buffer
        diff = l.difference(buf)
        if diff.is_empty:
            continue
        
        if diff.geom_type == "LineString":
            unmatched_segments.append(diff)
        elif diff.geom_type == "MultiLineString":
            unmatched_segments.extend(list(diff.geoms))
        elif diff.geom_type == "GeometryCollection":
            # Handle GeometryCollection which may contain LineStrings
            for geom in diff.geoms:
                if geom.geom_type == "LineString":
                    unmatched_segments.append(geom)
    
    return unmatched_segments

def evaluate(gt_path, pred_path, tol=1.0):
    """Evaluate using alternative file reading method"""
    
    # Create output directory and report file
    pred_filename = os.path.splitext(os.path.basename(pred_path))[0]
    output_dir = f"/wsl/eval_errors/{pred_filename}_diagnostics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create report file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
    
    # Redirect output to both console and file
    original_stdout = sys.stdout
    with open(report_path, 'w') as report_file:
        # Write header to file
        report_file.write(f"SHAPEFILE EVALUATION REPORT\n")
        report_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_file.write(f"Ground Truth: {gt_path}\n")
        report_file.write(f"Prediction: {pred_path}\n")
        report_file.write(f"Tolerance: {tol}\n")
        report_file.write("="*80 + "\n\n")
        
        # Set up tee output
        tee = TeeOutput(sys.stdout, report_file)
        sys.stdout = tee
        
        try:
            print("="*60)
            print(f"EVALUATING: {os.path.basename(pred_path)}")
            print(f"Ground Truth: {os.path.basename(gt_path)}")
            print(f"Tolerance: {tol}")
            print("="*60)
            
            try:
                # Try with GeoPandas first
                print("Attempting to read files with GeoPandas...")
                import geopandas as gpd
                gt_gdf = gpd.read_file(gt_path)
                pred_gdf = gpd.read_file(pred_path)
                gt_geometries = list(gt_gdf.geometry)
                pred_geometries = list(pred_gdf.geometry)
                print("✓ Successfully read files with GeoPandas")
            except Exception as e:
                print(f"✗ GeoPandas failed: {e}")
                print("Trying alternative method with fiona...")
                try:
                    gt_geometries = read_shapefile_alternative(gt_path)
                    pred_geometries = read_shapefile_alternative(pred_path)
                    print("✓ Successfully read files with fiona directly")
                except Exception as e2:
                    print(f"✗ Fiona also failed: {e2}")
                    raise e2

            print(f"GT geometries: {len(gt_geometries)}")
            print(f"Prediction geometries: {len(pred_geometries)}")

            gt_lines = flatten_lines(gt_geometries)
            pred_lines = flatten_lines(pred_geometries)

            print(f"GT lines: {len(gt_lines)}")
            print(f"Prediction lines: {len(pred_lines)}")

            print("Creating unions...")
            gt_union = unary_union(gt_lines) if gt_lines else unary_union([])
            pred_union = unary_union(pred_lines) if pred_lines else unary_union([])

            total_gt = total_length(gt_lines)
            total_pred = total_length(pred_lines)

            print(f"Total GT length: {total_gt:.2f}")
            print(f"Total prediction length: {total_pred:.2f}")

            matched_gt = matched_length(gt_lines, pred_union, tol)
            matched_pred = matched_length(pred_lines, gt_union, tol)

            recall = matched_gt / (total_gt + 1e-12)
            precision = matched_pred / (total_pred + 1e-12)
            f1 = 2*precision*recall / (precision+recall+1e-12)

            print(f"Matched GT length: {matched_gt:.2f} (Recall: {recall:.3f})")
            print(f"Matched Pred length: {matched_pred:.2f} (Precision: {precision:.3f})")
            print(f"F1 Score: {f1:.3f}")

            # Get matched and unmatched segments for visualization
            matched_gt_segments = get_matched_segments(gt_lines, pred_union, tol)
            unmatched_gt_segments = get_unmatched_segments(gt_lines, pred_union, tol)
            matched_pred_segments = get_matched_segments(pred_lines, gt_union, tol)
            unmatched_pred_segments = get_unmatched_segments(pred_lines, gt_union, tol)

            # intersections
            bad_inters_pred = bad_intersections(pred_lines)
            bad_inters_gt = bad_intersections(gt_lines)

            # dangling ends
            print("Analyzing network topology...")
            _, gt_endpoints, _ = build_graph(gt_lines)
            _, pred_endpoints, _ = build_graph(pred_lines)
            bad_dangling_pred = bad_dangling(pred_endpoints, gt_endpoints, 2*tol)
            
            print(f"Saving diagnostic files to: {output_dir}")

            # Export diagnostic layers to the named folder
            try:
                print("Saving shapefiles...")
                
                # Save matched and unmatched segments
                if matched_gt_segments:
                    matched_gt_gdf = gpd.GeoDataFrame({'type': ['matched_gt'] * len(matched_gt_segments)}, 
                                                    geometry=matched_gt_segments)
                    matched_gt_path = os.path.join(output_dir, "matched_gt_segments.shp")
                    matched_gt_gdf.to_file(matched_gt_path)
                    print(f"✓ Saved {len(matched_gt_segments)} matched GT segments")

                if unmatched_gt_segments:
                    unmatched_gt_gdf = gpd.GeoDataFrame({'type': ['unmatched_gt'] * len(unmatched_gt_segments)}, 
                                                      geometry=unmatched_gt_segments)
                    unmatched_gt_path = os.path.join(output_dir, "unmatched_gt_segments.shp")
                    unmatched_gt_gdf.to_file(unmatched_gt_path)
                    print(f"✓ Saved {len(unmatched_gt_segments)} unmatched GT segments")

                if matched_pred_segments:
                    matched_pred_gdf = gpd.GeoDataFrame({'type': ['matched_pred'] * len(matched_pred_segments)}, 
                                                      geometry=matched_pred_segments)
                    matched_pred_path = os.path.join(output_dir, "matched_pred_segments.shp")
                    matched_pred_gdf.to_file(matched_pred_path)
                    print(f"✓ Saved {len(matched_pred_segments)} matched prediction segments")

                if unmatched_pred_segments:
                    unmatched_pred_gdf = gpd.GeoDataFrame({'type': ['unmatched_pred'] * len(unmatched_pred_segments)}, 
                                                        geometry=unmatched_pred_segments)
                    unmatched_pred_path = os.path.join(output_dir, "unmatched_pred_segments.shp")
                    unmatched_pred_gdf.to_file(unmatched_pred_path)
                    print(f"✓ Saved {len(unmatched_pred_segments)} unmatched prediction segments")

                # Save intersections and dangling points
                if bad_inters_pred:
                    intersections_path = os.path.join(output_dir, "bad_intersections_pred.shp")
                    gpd.GeoDataFrame(geometry=bad_inters_pred).to_file(intersections_path)
                    print(f"✓ Saved {len(bad_inters_pred)} bad intersections")

                if bad_dangling_pred:
                    dangling_path = os.path.join(output_dir, "bad_dangling_pred.shp")
                    gpd.GeoDataFrame(geometry=bad_dangling_pred).to_file(dangling_path)
                    print(f"✓ Saved {len(bad_dangling_pred)} bad dangling points")
                    
                if bad_inters_gt:
                    gt_intersections_path = os.path.join(output_dir, "bad_intersections_gt.shp")
                    gpd.GeoDataFrame(geometry=bad_inters_gt).to_file(gt_intersections_path)
                    print(f"✓ Saved {len(bad_inters_gt)} bad intersections (GT)")
                    
            except Exception as e:
                print(f"✗ Warning: Could not save diagnostic shapefiles: {e}")
                import traceback
                traceback.print_exc()

            metrics = {
                "total_gt_length": total_gt,
                "total_pred_length": total_pred,
                "matched_gt_length": matched_gt,
                "matched_pred_length": matched_pred,
                "recall_length": recall,
                "precision_length": precision,
                "f1_length": f1,
                "n_bad_intersections_pred": len(bad_inters_pred),
                "n_bad_intersections_gt": len(bad_inters_gt),
                "n_bad_dangling_pred": len(bad_dangling_pred),
                "n_matched_gt_segments": len(matched_gt_segments),
                "n_unmatched_gt_segments": len(unmatched_gt_segments),
                "n_matched_pred_segments": len(matched_pred_segments),
                "n_unmatched_pred_segments": len(unmatched_pred_segments),
                "output_directory": output_dir,
                "report_file": report_path
            }
            
            print("="*60)
            print("EVALUATION COMPLETE!")
            print("="*60)
            
            print(f"\nFINAL METRICS:")
            print("-" * 40)
            for k,v in metrics.items():
                if k not in ['output_directory', 'report_file']:
                    if isinstance(v, float):
                        print(f"{k}: {v:.4f}")
                    else:
                        print(f"{k}: {v}")
            
            print(f"\nReport saved to: {report_path}")
            print(f"Shapefiles saved to: {output_dir}")
            
            # Write summary to file
            report_file.write(f"\n\nSUMMARY:\n")
            report_file.write("-" * 40 + "\n")
            for k,v in metrics.items():
                if k not in ['output_directory', 'report_file']:
                    if isinstance(v, float):
                        report_file.write(f"{k}: {v:.4f}\n")
                    else:
                        report_file.write(f"{k}: {v}\n")
            
            return metrics
            
        finally:
            # Restore original stdout
            sys.stdout = original_stdout

if __name__ == "__main__":
    import glob
    
    gt_file = "/wsl/eval_errors/GT_TM10_test.shp"
    pred_dir = "/wsl/eval_errors/test_TM10"  # Directory containing prediction shapefiles
    
    ## 6.408 = 2.563
    ## 4.272 = 1.708
    ## 2.136 = 0.854
    tol = 0.854

    # Check if ground truth file exists
    if not os.path.exists(gt_file):
        print(f"Ground truth file not found: {gt_file}")
        exit(1)
    
    # Find all shapefiles in the prediction directory
    if os.path.isdir(pred_dir):
        pred_files = glob.glob(os.path.join(pred_dir, "*.shp"))
        if not pred_files:
            print(f"No shapefiles found in directory: {pred_dir}")
            exit(1)
        print(f"Found {len(pred_files)} shapefiles to evaluate")
    elif os.path.isfile(pred_dir) and pred_dir.endswith('.shp'):
        # Single file mode
        pred_files = [pred_dir]
    else:
        print(f"Invalid prediction path: {pred_dir}")
        exit(1)
    
    # Store all results
    all_results = []
    failed_files = []
    
    print("="*80)
    print(f"BATCH EVALUATION OF {len(pred_files)} SHAPEFILES")
    print(f"Ground Truth: {gt_file}")
    print(f"Tolerance: {tol}")
    print("="*80)
    print()
    
    # Evaluate each shapefile
    for idx, pred_file in enumerate(pred_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing {idx}/{len(pred_files)}: {os.path.basename(pred_file)}")
        print(f"{'='*80}\n")
        
        try:
            metrics = evaluate(gt_file, pred_file, tol=tol)
            all_results.append({
                'file': os.path.basename(pred_file),
                'metrics': metrics
            })
            print(f"\n✓ Successfully evaluated {os.path.basename(pred_file)}")
            
        except Exception as e:
            print(f"\n✗ Error evaluating {os.path.basename(pred_file)}: {e}")
            import traceback
            traceback.print_exc()
            failed_files.append({
                'file': os.path.basename(pred_file),
                'error': str(e)
            })
    
    # Create summary report
    print(f"\n\n{'='*80}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Total files processed: {len(pred_files)}")
    print(f"Successfully evaluated: {len(all_results)}")
    print(f"Failed: {len(failed_files)}")
    print()
    
    if all_results:
        print("RESULTS SUMMARY:")
        print("-" * 80)
        print(f"{'File':<40} {'Recall':>10} {'Precision':>10} {'F1':>10}")
        print("-" * 80)
        
        for result in all_results:
            filename = result['file']
            metrics = result['metrics']
            print(f"{filename:<40} {metrics['recall_length']:>10.4f} "
                  f"{metrics['precision_length']:>10.4f} {metrics['f1_length']:>10.4f}")
        
        print("-" * 80)
        
        # Calculate averages
        avg_recall = sum(r['metrics']['recall_length'] for r in all_results) / len(all_results)
        avg_precision = sum(r['metrics']['precision_length'] for r in all_results) / len(all_results)
        avg_f1 = sum(r['metrics']['f1_length'] for r in all_results) / len(all_results)
        
        print(f"{'AVERAGE':<40} {avg_recall:>10.4f} {avg_precision:>10.4f} {avg_f1:>10.4f}")
        print("-" * 80)
        print()
    
    if failed_files:
        print("FAILED FILES:")
        print("-" * 80)
        for failed in failed_files:
            print(f"  {failed['file']}: {failed['error']}")
        print()
    
    # Save consolidated summary report
    summary_path = os.path.join(pred_dir, "batch_evaluation_summary.txt")
    try:
        with open(summary_path, 'w') as f:
            f.write("BATCH SHAPEFILE EVALUATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ground Truth: {gt_file}\n")
            f.write(f"Prediction Directory: {pred_dir}\n")
            f.write(f"Tolerance: {tol}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total files processed: {len(pred_files)}\n")
            f.write(f"Successfully evaluated: {len(all_results)}\n")
            f.write(f"Failed: {len(failed_files)}\n\n")
            
            if all_results:
                f.write("RESULTS SUMMARY:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'File':<40} {'Recall':>10} {'Precision':>10} {'F1':>10}\n")
                f.write("-" * 80 + "\n")
                
                for result in all_results:
                    filename = result['file']
                    metrics = result['metrics']
                    f.write(f"{filename:<40} {metrics['recall_length']:>10.4f} "
                           f"{metrics['precision_length']:>10.4f} {metrics['f1_length']:>10.4f}\n")
                
                f.write("-" * 80 + "\n")
                f.write(f"{'AVERAGE':<40} {avg_recall:>10.4f} {avg_precision:>10.4f} {avg_f1:>10.4f}\n")
                f.write("-" * 80 + "\n\n")
                
                # Detailed metrics
                f.write("\nDETAILED METRICS:\n")
                f.write("="*80 + "\n\n")
                for result in all_results:
                    f.write(f"\n{result['file']}:\n")
                    f.write("-" * 40 + "\n")
                    for k, v in result['metrics'].items():
                        if k not in ['output_directory', 'report_file']:
                            if isinstance(v, float):
                                f.write(f"  {k}: {v:.4f}\n")
                            else:
                                f.write(f"  {k}: {v}\n")
                    f.write(f"  Report: {result['metrics']['report_file']}\n")
                    f.write(f"  Diagnostics: {result['metrics']['output_directory']}\n")
                    f.write("\n")
            
            if failed_files:
                f.write("\nFAILED FILES:\n")
                f.write("-" * 80 + "\n")
                for failed in failed_files:
                    f.write(f"  {failed['file']}: {failed['error']}\n")
        
        print(f"Summary report saved to: {summary_path}")
        
    except Exception as e:
        print(f"Warning: Could not save summary report: {e}")
    
    # Save results as JSON for programmatic access
    json_path = os.path.join(pred_dir, "batch_evaluation_results.json")
    try:
        import json
        json_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ground_truth': gt_file,
            'prediction_directory': pred_dir,
            'tolerance': tol,
            'total_files': len(pred_files),
            'successful': len(all_results),
            'failed': len(failed_files),
            'results': all_results,
            'failed_files': failed_files
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON results saved to: {json_path}")
    except Exception as e:
        print(f"Warning: Could not save JSON results: {e}")
    
    print(f"\n{'='*80}")
    print("BATCH EVALUATION COMPLETE!")
    print(f"{'='*80}")
