import json
import os
import pandas as pd
import time
import requests
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import ADDITIONAL_DATA_PATH, COMBINED_INFORMATION_CSV_PATH
from helpers.file_manager import FileManager
from helpers.prompts.artist_prompt import ArtistPrompt
from helpers.prompts.auction_house_prompt import AuctionHousePrompt
from helpers.prompts.base_prompt import BasePrompt
from helpers.property_extractor import PropertyExtractor


class SyntheticDataGenerator:
    FILES_API_URL = "https://api.openai.com/v1/files"
    BATCH_API_URL = "https://api.openai.com/v1/batches"
    COMPLETION_ENDPOINT = "/v1/chat/completions"

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.1,
        use_batch: bool = False,
        pending_batches_file: str = "pending_batches.txt",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.use_batch = use_batch
        self.pending_batches_file = pending_batches_file

        self.model = ChatOpenAI(
            model=model_name, openai_api_key=self.api_key, temperature=temperature
        )

    def generate_batch_local(
        self, data: pd.DataFrame, prompt_template: ChatPromptTemplate
    ) -> List[str]:
        filled_prompts = [
            prompt_template.format_messages(**row.to_dict())
            for _, row in data.iterrows()
        ]
        responses = self.model.batch(filled_prompts)
        return [r.content for r in responses]

    def generate_batch_openai(
        self, data: pd.DataFrame, prompt_template: ChatPromptTemplate
    ) -> str:
        jsonl_file = self._prepare_jsonl_file(data, prompt_template)
        file_id = self._upload_file(jsonl_file)
        batch_id = self._submit_batch_job(file_id)
        self._save_batch_id(batch_id)
        return batch_id

    def _prepare_jsonl_file(
        self, data: pd.DataFrame, prompt_template: ChatPromptTemplate
    ) -> str:
        jsonl_file = f"batch_request_{int(time.time())}.jsonl"
        with open(jsonl_file, "w") as f:
            for idx, row in data.iterrows():
                filled_prompt = prompt_template.format_messages(**row.to_dict())

                request_object = {
                    "custom_id": f"row_{idx}",
                    "method": "POST",
                    "url": self.COMPLETION_ENDPOINT,
                    "body": {
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "messages": [msg.dict() for msg in filled_prompt],
                    },
                }

                f.write(json.dumps(request_object) + "\n")
        return jsonl_file

    def _upload_file(self, file_path: str) -> str:
        response = requests.post(
            self.FILES_API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": open(file_path, "rb")},
            data={"purpose": "batch"},
        )
        response.raise_for_status()
        return response.json()["id"]

    def _submit_batch_job(self, file_id: str) -> str:
        payload = {
            "input_file_id": file_id,
            "endpoint": self.COMPLETION_ENDPOINT,
            "completion_window": "24h",
        }
        response = requests.post(
            self.BATCH_API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return response.json()["id"]

    def _save_batch_id(self, batch_id: str):
        with open(self.pending_batches_file, "a") as f:
            f.write(batch_id + "\n")

    def clean_and_parse_json(self, response: str) -> dict:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        response = response.strip("`").strip()
        return json.loads(response)

    def try_enrich_dataframe(
        self, df: pd.DataFrame, prompt_class: BasePrompt
    ) -> Optional[pd.DataFrame]:
        prompt_template = prompt_class.get_prompt_template()

        if self.use_batch:
            batch_id = self.generate_batch_openai(df, prompt_template)
            print(f"Batch submitted: {batch_id} (Saved to {self.PENDING_BATCHES_FILE})")
            return pd.DataFrame()  # Return empty DataFrame
        else:
            synthetic_data = self.generate_batch_local(df, prompt_template)

            parsed_data = [self.clean_and_parse_json(entry) for entry in synthetic_data]
            synthetic_df = pd.DataFrame(parsed_data)

            synthetic_df["Score Explanations"] = synthetic_df[
                "Score Explanations"
            ].apply(lambda d: "; ".join([f"{k}: {v}" for k, v in d.items()]))

            return pd.concat([df, synthetic_df], axis=1)

    def fetch_completed_batches(self, output_folder="completed_batches"):
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(self.PENDING_BATCHES_FILE):
            print("No pending batches to fetch.")
            return

        with open(self.PENDING_BATCHES_FILE, "r") as f:
            batch_ids = [line.strip() for line in f.readlines()]

        still_pending = []

        for batch_id in batch_ids:
            response = requests.get(
                f"{self.BATCH_API_URL}/{batch_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            status = response.json()["status"]

            if status == "completed":
                output_file_id = response.json()["output_file_id"]
                self._download_and_save_result(output_file_id, output_folder, batch_id)
            elif status in ["failed", "cancelled"]:
                print(f"Batch {batch_id} failed or was cancelled.")
            else:
                still_pending.append(batch_id)

        with open(self.PENDING_BATCHES_FILE, "w") as f:
            for pending_id in still_pending:
                f.write(pending_id + "\n")

    def _download_and_save_result(
        self, file_id: str, output_folder: str, batch_id: str
    ):
        response = requests.get(
            f"{self.FILES_API_URL}/{file_id}/content",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()

        output_path = os.path.join(output_folder, f"batch_{batch_id}.jsonl")
        with open(output_path, "w") as f:
            f.write(response.text)

        print(f"Downloaded results for batch {batch_id} to {output_path}")


if __name__ == "__main__":
    pending_batch_file_path = f"{ADDITIONAL_DATA_PATH}/pending_batches.txt"

    data_to_enrich = FileManager.read_single_csv(COMBINED_INFORMATION_CSV_PATH)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    generator = SyntheticDataGenerator(
        model_name="gpt-4o", temperature=0, use_batch=True
    )

    # artists = PropertyExtractor.get_unique_dataframe(data_to_enrich, "Artist name")[:5]
    # artists_enriched = generator.try_enrich_dataframe(artists, ArtistPrompt())
    # artists_enriched.to_csv(f"{ADDITIONAL_DATA_PATH}/artists_{timestamp}.csv", sep=";")

    auction_houses = PropertyExtractor.get_unique_dataframe(
        data_to_enrich, "Auction House"
    )[:5]
    auction_houses_enriched = generator.try_enrich_dataframe(
        auction_houses, AuctionHousePrompt()
    )
    auction_houses_enriched.to_csv(
        f"{ADDITIONAL_DATA_PATH}/auction_houses_{timestamp}.csv", sep=";"
    )
