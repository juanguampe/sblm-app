# LanceDB Database

This directory should contain the LanceDB database files.

## Important note for Streamlit Cloud deployment

For Streamlit Cloud deployment, you'll need to upload your database files directly through the Streamlit interface.

1. Go to your app in Streamlit Cloud
2. Click on "Manage app" in the dropdown menu at the top right
3. Go to "Settings" tab
4. Under "App dependencies", click on "Upload files"
5. Upload the entire `docling.lance` directory from your local machine

This step is required because your vector database contains binary files that can't be efficiently stored in a GitHub repository.