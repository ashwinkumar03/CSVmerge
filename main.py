from flask import Flask, request, send_file, render_template, flash, redirect, url_for
import pandas as pd
import os
import re
from typing import Optional, List

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flashing messages

# Configuration
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

# Column mapping dictionary
COLUMN_MAPPING = {
    "Employee Range": "Headcount",
    "Domain": "Website",
    "Linkedin Url": "LinkedIn",
    "Founded": "Year Founded",
    "Employees on Professional Networks": "Employees Count (LinkedIn)",
    "Est. Revenue (MUSD) (MAX 1B)": "Estimated Revenue",
    "Revenue Estimate": "Estimated Revenue",
    "Employee Growth (Annual)": "Headcount Growth (12 Months)",
    "Ownership Type": "Ownership",
}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def standardize_url(url: str) -> str:
    """Standardize URL format by removing http(s):// and www."""
    if pd.notna(url):
        return re.sub(r'^https?://(www\.)?', '', str(url))
    return url

def is_column_empty(series: pd.Series) -> bool:
    """Check if a column is effectively empty after cleaning."""
    cleaned_data = series.str.replace(r'File[123]: \s*\n*', '', regex=True).str.strip()
    return cleaned_data.replace('', pd.NA).isna().all()

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process a single dataframe with standard cleaning operations."""
    if df is None:
        return None

    # Rename columns according to mapping
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    # Clean data
    df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else '')

    # Standardize Website URLs if present
    if 'Website' in df.columns:
        df['Website'] = df['Website'].apply(standardize_url)

    return df

def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple dataframes with proper handling of overlapping columns."""
    if not dfs or not any(df is not None for df in dfs):
        raise ValueError("No valid dataframes provided for merging")

    # Filter out None values and get the first valid dataframe
    valid_dfs = [df for df in dfs if df is not None]
    result = valid_dfs[0]

    # Merge subsequent dataframes
    for i, df in enumerate(valid_dfs[1:], 2):
        result = result.merge(df, on="Website", how="outer", 
                            suffixes=(f'_{i-1}', f'_{i}'))

    # Process and combine overlapping columns
    for col in result.columns:
        if any(f'_{i}' in col for i in range(1, len(valid_dfs) + 1)):
            base_col = col.split('_')[0]
            file_num = col.split('_')[-1]

            mask = result[col].str.strip() != ''
            if mask.any():
                prefix_data = f'File{file_num}: ' + result[col]
                if base_col not in result:
                    result[base_col] = ''

                result.loc[mask, base_col] = (
                    result.loc[mask, base_col].str.strip().replace('', pd.NA).fillna('') +
                    ('\n\n' if result.loc[mask, base_col].str.strip().any() else '') +
                    prefix_data[mask]
                )

    # Clean up the merged dataframe
    result = result.loc[:, ~result.columns.str.endswith(tuple(f'_{i}' for i in range(1, len(valid_dfs) + 1)))]
    columns_to_keep = [col for col in result.columns if not is_column_empty(result[col])]

    return result[columns_to_keep]

@app.route('/')
def upload_form():
    """Render the upload form."""
    return render_template('upload.html')

@app.route('/merge', methods=['POST'])
def merge_csvs():
    """Handle file uploads and CSV merging."""
    try:
        # Validate file uploads
        if 'file1' not in request.files or 'file2' not in request.files:
            flash('Please upload at least two CSV files.', 'error')
            return redirect(url_for('upload_form'))

        files = [request.files['file1'], request.files['file2']]
        if 'file3' in request.files:
            files.append(request.files['file3'])

        # Validate each file
        for i, file in enumerate(files, 1):
            if not file or not file.filename:
                if i <= 2:  # First two files are required
                    return f'File {i} is required.'
                continue

            if not allowed_file(file.filename):
                return f'File {i} must be a CSV file.'

        # Read and process dataframes
        dfs = []
        for file in files:
            if file and file.filename:
                try:
                    df = pd.read_csv(file)
                    df = process_dataframe(df)
                    dfs.append(df)
                except Exception as e:
                    return f'Error reading file {file.filename}: {str(e)}'

        # Merge dataframes
        merged_df = merge_dataframes(dfs)

        # Save and send result
        output_file = "merged_output.csv"
        merged_df.to_csv(output_file, index=False)

        return send_file(
            output_file,
            as_attachment=True,
            mimetype='text/csv'
        )

    except Exception as e:
        return f'An error occurred: {str(e)}'

    finally:
        # Cleanup temporary files
        if 'output_file' in locals() and os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)