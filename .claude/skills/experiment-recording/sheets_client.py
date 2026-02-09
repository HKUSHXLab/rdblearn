"""
Google Sheets API client for experiment recording.
"""

import gspread
import pandas as pd
from pathlib import Path
from typing import Optional, List

from constants import SPREADSHEET_ID, MAIN_SHEET_NAME


class SheetsClient:
    """Client for interacting with Google Sheets."""

    def __init__(self, spreadsheet_id: str = SPREADSHEET_ID):
        """
        Initialize Google Sheets client.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID (from URL)
        """
        self.spreadsheet_id = spreadsheet_id
        self._client = None
        self._spreadsheet = None

        # Load credentials from same directory as this file
        self.credentials_path = Path(__file__).parent / "credentials.json"

    def connect(self):
        """Establish connection to Google Sheets."""
        if not self.credentials_path.exists():
            raise FileNotFoundError(
                f"Credentials file not found: {self.credentials_path}\n"
                "Please ensure credentials.json is in the skill folder."
            )

        self._client = gspread.service_account(filename=str(self.credentials_path))
        self._spreadsheet = self._client.open_by_key(self.spreadsheet_id)

        print(f"Connected to spreadsheet: {self._spreadsheet.title}")
        return self

    def read_target_table(self, sheet_name: str = MAIN_SHEET_NAME) -> pd.DataFrame:
        """
        Read the target table from a sheet.

        Args:
            sheet_name: Name of the worksheet to read

        Returns:
            DataFrame with target table data
        """
        try:
            worksheet = self._spreadsheet.worksheet(sheet_name)
            all_values = worksheet.get_all_values()
            if not all_values or len(all_values) < 2:
                return pd.DataFrame()
            headers = all_values[0]
            # Filter out columns with empty headers
            valid_cols = [i for i, h in enumerate(headers) if h.strip()]
            filtered_headers = [headers[i] for i in valid_cols]
            filtered_rows = [[row[i] if i < len(row) else "" for i in valid_cols] for row in all_values[1:]]
            df = pd.DataFrame(filtered_rows, columns=filtered_headers)
            # Remove fully empty rows
            df = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
            # Convert numeric columns from strings to numbers
            for col in df.columns:
                converted = pd.to_numeric(df[col], errors='coerce')
                # Only apply conversion if at least some values converted successfully
                if converted.notna().any():
                    # Keep original strings where conversion failed (non-numeric columns)
                    if converted.isna().all():
                        continue
                    # If most values are numeric, treat as numeric column
                    if converted.notna().sum() > converted.isna().sum():
                        df[col] = converted
            return df
        except gspread.WorksheetNotFound:
            print(f"Sheet '{sheet_name}' not found")
            return pd.DataFrame()

    def write_comparison_sheet(
        self,
        sheet_name: str,
        comparison_df: pd.DataFrame,
        improvement_rate: float,
        r2_target_df: Optional[pd.DataFrame] = None,
        detailed_df: Optional[pd.DataFrame] = None,
        sweep_id: Optional[str] = None
    ):
        """
        Write comparison table to a new sheet, with optional R² summary and detailed table below.

        Args:
            sheet_name: Name for the new worksheet
            comparison_df: DataFrame with comparison data
            improvement_rate: Improvement rate percentage
            r2_target_df: Optional DataFrame with R²-based target table (regression tasks)
            detailed_df: Optional DataFrame with all runs (detailed table)
            sweep_id: Optional sweep ID for traceability
        """
        # Calculate total rows needed
        total_rows = len(comparison_df) + 10
        if r2_target_df is not None and not r2_target_df.empty:
            total_rows += len(r2_target_df) + 5
        if detailed_df is not None:
            total_rows += len(detailed_df) + 5

        # Create new worksheet
        try:
            worksheet = self._spreadsheet.add_worksheet(
                title=sheet_name,
                rows=total_rows,
                cols=max(len(comparison_df.columns), len(detailed_df.columns) if detailed_df is not None else 0) + 2
            )
        except gspread.exceptions.APIError as e:
            if "already exists" in str(e):
                # Delete and recreate
                self._spreadsheet.del_worksheet(
                    self._spreadsheet.worksheet(sheet_name)
                )
                worksheet = self._spreadsheet.add_worksheet(
                    title=sheet_name,
                    rows=total_rows,
                    cols=max(len(comparison_df.columns), len(detailed_df.columns) if detailed_df is not None else 0) + 2
                )
            else:
                raise

        # Prepare data for writing
        headers = list(comparison_df.columns)
        rows = [headers]

        for _, row in comparison_df.iterrows():
            row_values = []
            for val in row:
                if pd.isna(val):
                    row_values.append("")
                elif isinstance(val, float):
                    row_values.append(round(val, 6))
                else:
                    row_values.append(val)
            rows.append(row_values)

        # Add improvement rate row
        rows.append([])  # Empty row
        improvement_row = [""] * (len(headers) - 1) + [f"Improvement Rate: {improvement_rate:.1f}%"]
        rows.append(improvement_row)

        # Add R² target table if provided (2 rows below improvement rate)
        r2_start_row = None
        if r2_target_df is not None and not r2_target_df.empty:
            rows.append([])  # Empty row
            rows.append([])  # Another empty row for spacing
            rows.append(["R² SUMMARY TABLE (Best dev_r2 per Regression Task)"])  # Section header
            r2_start_row = len(rows) + 1  # 1-indexed for sheets

            # Add R² table headers
            r2_headers = list(r2_target_df.columns)
            rows.append(r2_headers)

            # Add R² table data
            for _, row in r2_target_df.iterrows():
                row_values = []
                for val in row:
                    if pd.isna(val):
                        row_values.append("")
                    elif isinstance(val, float):
                        row_values.append(round(val, 6))
                    else:
                        row_values.append(val)
                rows.append(row_values)

        # Add sweep URL for traceability
        if sweep_id:
            rows.append([])  # Empty row
            sweep_url = f"https://wandb.ai/tgif/rdblearn-scripts/sweeps/{sweep_id}"
            rows.append([f"Source Sweep: {sweep_url}"])

        # Add detailed table if provided
        detailed_start_row = None
        if detailed_df is not None and not detailed_df.empty:
            rows.append([])  # Empty row
            rows.append([])  # Another empty row for spacing
            rows.append(["DETAILED TABLE (All Runs)"])  # Section header
            detailed_start_row = len(rows) + 1  # 1-indexed for sheets

            # Add detailed table headers
            detailed_headers = list(detailed_df.columns)
            rows.append(detailed_headers)

            # Add detailed table data
            for _, row in detailed_df.iterrows():
                row_values = []
                for val in row:
                    if pd.isna(val):
                        row_values.append("")
                    elif isinstance(val, float):
                        row_values.append(round(val, 6))
                    else:
                        row_values.append(val)
                rows.append(row_values)

        # Write all data at once
        worksheet.update(rows, 'A1')

        # Format comparison table header row (bold with blue background)
        worksheet.format('A1:' + chr(ord('A') + len(headers) - 1) + '1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.27, 'green': 0.45, 'blue': 0.77}
        })

        # Apply conditional formatting for delta_test column
        delta_col_idx = headers.index('delta_test') if 'delta_test' in headers else None
        if delta_col_idx is not None:
            delta_col_letter = chr(ord('A') + delta_col_idx)
            # Data rows: from row 2 to row (1 + len(comparison_df))
            data_end_row = 1 + len(comparison_df)
            delta_range = f"{delta_col_letter}2:{delta_col_letter}{data_end_row}"

            # Apply green for positive (improvement when direction is 'up')
            # Apply red for negative (degradation when direction is 'up')
            # Note: We need to check direction per row, so we apply formatting cell by cell
            self._apply_delta_formatting(worksheet, comparison_df, delta_col_idx)

        # Format R² table header if present
        if r2_start_row is not None and r2_target_df is not None:
            r2_headers = list(r2_target_df.columns)
            # Format section title
            worksheet.format(f'A{r2_start_row - 1}', {
                'textFormat': {'bold': True, 'fontSize': 12}
            })
            # Format R² table header row (green background for R² theme)
            worksheet.format(
                f'A{r2_start_row}:' + chr(ord('A') + len(r2_headers) - 1) + str(r2_start_row),
                {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.56, 'green': 0.77, 'blue': 0.49}
                }
            )

        # Format detailed table header if present
        if detailed_start_row is not None and detailed_df is not None:
            detailed_headers = list(detailed_df.columns)
            # Format section title
            worksheet.format(f'A{detailed_start_row - 1}', {
                'textFormat': {'bold': True, 'fontSize': 12}
            })
            # Format detailed table header row
            worksheet.format(
                f'A{detailed_start_row}:' + chr(ord('A') + len(detailed_headers) - 1) + str(detailed_start_row),
                {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.6, 'green': 0.6, 'blue': 0.6}
                }
            )

        print(f"Comparison table written to sheet: {sheet_name}")
        return worksheet

    def _apply_delta_formatting(
        self,
        worksheet,
        comparison_df: pd.DataFrame,
        delta_col_idx: int
    ):
        """
        Apply conditional formatting to delta_test column based on direction.

        Green = improvement (positive delta when direction='up', negative when direction='down')
        Red = degradation (negative delta when direction='up', positive when direction='down')

        Args:
            worksheet: The gspread worksheet
            comparison_df: DataFrame with comparison data
            delta_col_idx: Index of delta_test column
        """
        delta_col_letter = chr(ord('A') + delta_col_idx)

        # Get direction column index
        headers = list(comparison_df.columns)
        direction_col_idx = headers.index('Direction') if 'Direction' in headers else None

        if direction_col_idx is None:
            return

        # Collect cells to format
        green_cells = []
        red_cells = []

        for row_idx, (_, row) in enumerate(comparison_df.iterrows(), start=2):  # Start at row 2 (after header)
            delta = row.get('delta_test')
            direction = row.get('Direction')

            if pd.isna(delta) or delta is None:
                continue

            # Determine if this is an improvement or degradation
            is_improvement = (direction == 'up' and delta > 0) or (direction == 'down' and delta < 0)
            is_degradation = (direction == 'up' and delta < 0) or (direction == 'down' and delta > 0)

            cell = f"{delta_col_letter}{row_idx}"
            if is_improvement:
                green_cells.append(cell)
            elif is_degradation:
                red_cells.append(cell)

        # Apply green formatting (light green background)
        if green_cells:
            for cell in green_cells:
                worksheet.format(cell, {
                    'backgroundColor': {'red': 0.72, 'green': 0.88, 'blue': 0.72}
                })

        # Apply red formatting (light red background)
        if red_cells:
            for cell in red_cells:
                worksheet.format(cell, {
                    'backgroundColor': {'red': 0.96, 'green': 0.72, 'blue': 0.72}
                })

    def update_main_sheet(
        self,
        target_df: pd.DataFrame,
        detailed_df: Optional[pd.DataFrame] = None,
        sweep_id: Optional[str] = None
    ):
        """
        Update the Main baseline sheet with new target table.

        Args:
            target_df: DataFrame with target table data
            detailed_df: Optional DataFrame with all runs (detailed table)
            sweep_id: Optional sweep ID for traceability
        """
        # Calculate total rows needed
        total_rows = len(target_df) + 10
        if detailed_df is not None:
            total_rows += len(detailed_df) + 5

        try:
            worksheet = self._spreadsheet.worksheet(MAIN_SHEET_NAME)
            worksheet.clear()
            # Resize if needed
            worksheet.resize(rows=total_rows, cols=max(
                len(target_df.columns),
                len(detailed_df.columns) if detailed_df is not None else 0
            ) + 2)
        except gspread.WorksheetNotFound:
            worksheet = self._spreadsheet.add_worksheet(
                title=MAIN_SHEET_NAME,
                rows=total_rows,
                cols=max(
                    len(target_df.columns),
                    len(detailed_df.columns) if detailed_df is not None else 0
                ) + 2
            )

        # Prepare data
        headers = list(target_df.columns)
        rows = [headers]

        for _, row in target_df.iterrows():
            row_values = []
            for val in row:
                if pd.isna(val):
                    row_values.append("")
                elif isinstance(val, float):
                    row_values.append(round(val, 6))
                else:
                    row_values.append(val)
            rows.append(row_values)

        # Add sweep URL for traceability
        if sweep_id:
            rows.append([])  # Empty row
            sweep_url = f"https://wandb.ai/tgif/rdblearn-scripts/sweeps/{sweep_id}"
            rows.append([f"Source Sweep: {sweep_url}"])

        # Add detailed table if provided
        detailed_start_row = None
        if detailed_df is not None and not detailed_df.empty:
            rows.append([])  # Empty row
            rows.append([])  # Another empty row for spacing
            rows.append(["DETAILED TABLE (All Runs)"])  # Section header
            detailed_start_row = len(rows) + 1  # 1-indexed for sheets

            # Add detailed table headers
            detailed_headers = list(detailed_df.columns)
            rows.append(detailed_headers)

            # Add detailed table data
            for _, row in detailed_df.iterrows():
                row_values = []
                for val in row:
                    if pd.isna(val):
                        row_values.append("")
                    elif isinstance(val, float):
                        row_values.append(round(val, 6))
                    else:
                        row_values.append(val)
                rows.append(row_values)

        # Write all data at once
        worksheet.update(rows, 'A1')

        # Format header row
        worksheet.format('A1:' + chr(ord('A') + len(headers) - 1) + '1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.27, 'green': 0.45, 'blue': 0.77}
        })

        # Format detailed table header if present
        if detailed_start_row is not None and detailed_df is not None:
            detailed_headers = list(detailed_df.columns)
            # Format section title
            worksheet.format(f'A{detailed_start_row - 1}', {
                'textFormat': {'bold': True, 'fontSize': 12}
            })
            # Format detailed table header row
            worksheet.format(
                f'A{detailed_start_row}:' + chr(ord('A') + len(detailed_headers) - 1) + str(detailed_start_row),
                {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.6, 'green': 0.6, 'blue': 0.6}
                }
            )

        print(f"Main sheet '{MAIN_SHEET_NAME}' updated with {len(target_df)} rows")

    def get_sheet_url(self, sheet_name: str) -> str:
        """Get the URL to a specific sheet."""
        try:
            worksheet = self._spreadsheet.worksheet(sheet_name)
            return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/edit#gid={worksheet.id}"
        except gspread.WorksheetNotFound:
            return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}"
