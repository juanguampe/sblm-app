import os
import zipfile
import sys

def zip_directory(source_dir, output_filename):
    """
    Create a zip file from a directory.
    
    Args:
        source_dir: Directory to be zipped
        output_filename: Name of the output zip file
    """
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                print(f"Adding {arcname}")
                zipf.write(file_path, arcname)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bundle_database.py <path_to_database>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    if not os.path.exists(db_path):
        print(f"Error: Path {db_path} does not exist.")
        sys.exit(1)
    
    output_file = "lancedb.zip"
    zip_directory(db_path, output_file)
    print(f"\nDatabase successfully bundled to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print("\nNext steps:")
    print("1. Upload this ZIP file to Google Drive, Dropbox, or another file sharing service")
    print("2. Make it publicly accessible and copy the download link")
    print("3. Update the app's download_database function with this link")
