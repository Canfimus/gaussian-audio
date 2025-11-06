"""
Debug script to find .npy files
"""
import os
import glob
import sys

if len(sys.argv) < 2:
    print("Usage: python debug_find_npy.py <directory>")
    print("Example: python debug_find_npy.py ./checkpoints/ljspeech_subset/GaussianImage_Cholesky_10000_50000")
    sys.exit(1)

search_dir = sys.argv[1]

print("="*70)
print("Debugging .npy file search")
print("="*70)
print(f"Search directory: {search_dir}")
print()

# Check if directory exists
if not os.path.exists(search_dir):
    print(f"❌ Directory does not exist: {search_dir}")
    sys.exit(1)

print("✅ Directory exists")
print()

# List immediate contents
print("Contents of directory (first level):")
print("-"*70)
try:
    contents = os.listdir(search_dir)
    for item in sorted(contents)[:20]:  # Show first 20 items
        item_path = os.path.join(search_dir, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")
    
    if len(contents) > 20:
        print(f"  ... and {len(contents) - 20} more items")
except Exception as e:
    print(f"❌ Error listing directory: {e}")
    sys.exit(1)

print()

# Try different glob patterns
print("Searching for .npy files with different patterns:")
print("-"*70)

# Pattern 1: Direct children
pattern1 = os.path.join(search_dir, '*_fitting.npy')
results1 = glob.glob(pattern1)
print(f"Pattern: {pattern1}")
print(f"Found: {len(results1)} files")
if results1:
    for f in results1[:5]:
        print(f"  • {f}")
print()

# Pattern 2: One level deep
pattern2 = os.path.join(search_dir, '*', '*_fitting.npy')
results2 = glob.glob(pattern2)
print(f"Pattern: {pattern2}")
print(f"Found: {len(results2)} files")
if results2:
    for f in results2[:5]:
        print(f"  • {f}")
print()

# Pattern 3: Recursive (requires Python 3.5+)
pattern3 = os.path.join(search_dir, '**', '*_fitting.npy')
results3 = glob.glob(pattern3, recursive=True)
print(f"Pattern: {pattern3} (recursive=True)")
print(f"Found: {len(results3)} files")
if results3:
    for f in results3[:5]:
        print(f"  • {f}")
print()

# Pattern 4: Any .npy file
pattern4 = os.path.join(search_dir, '**', '*.npy')
results4 = glob.glob(pattern4, recursive=True)
print(f"Pattern: {pattern4} (recursive=True)")
print(f"Found: {len(results4)} files")
if results4:
    for f in results4[:10]:
        print(f"  • {f}")
print()

# Summary
print("="*70)
print("SUMMARY")
print("="*70)
if results3:
    print(f"✅ Found {len(results3)} *_fitting.npy files")
    print()
    print("The script SHOULD work. The correct path to use is:")
    print(f"  {search_dir}")
elif results4:
    print(f"⚠️  Found {len(results4)} .npy files, but none matching '*_fitting.npy'")
    print()
    print("Possible issues:")
    print("  1. Files are not named with '_fitting.npy' suffix")
    print("  2. Files are in a different location")
else:
    print("❌ No .npy files found at all!")
    print()
    print("Possible issues:")
    print("  1. Wrong directory")
    print("  2. Files not saved (did you use --save_imgs flag?)")
    print("  3. Different directory structure than expected")

print("="*70)