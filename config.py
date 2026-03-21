import os
import csv
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

DATA_SINK_TEST = 'test'


def create_supabase_client() -> Client:
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    
    return create_client(url, key)


class DataSink:
    def __init__(self, mode: str | None = None):
        self.mode = mode or 'actual'
        self.supabase_client = create_supabase_client() if self.mode == 'actual' else None

    async def submit_rows(self, table_name, rows_data, csv_filename=None, fieldnames=None):
        if not rows_data:
            return True

        try:
            if self.mode == 'test':
                self._write_csv(rows_data, csv_filename or f'{table_name}_test.csv', fieldnames)
            else:
                self.supabase_client.table(table_name).insert(rows_data).execute()
            return True
        except Exception as e:
            sink_name = 'CSV' if self.mode == 'test' else 'Supabase'
            print(f'{sink_name} write failed for {table_name}: {type(e).__name__}: {e}')
            return False

    def _write_csv(self, rows_data, csv_filename, fieldnames=None):
        resolved_fieldnames = list(fieldnames or rows_data[0].keys())
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=resolved_fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerows(rows_data)


def create_data_sink(mode: str | None = None) -> DataSink:
    return DataSink(mode=mode)
