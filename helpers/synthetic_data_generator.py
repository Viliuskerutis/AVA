import json
import os
import pandas as pd
import time
import requests
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import ADDITIONAL_DATA_PATH, CSVS_PATH
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
        file_id = self._prepare_and_upload_jsonl(data, prompt_template)
        batch_id = self._submit_batch_job(file_id)
        self._save_batch_id(batch_id)
        return batch_id

    def _prepare_and_upload_jsonl(
        self, data: pd.DataFrame, prompt_template: ChatPromptTemplate
    ) -> str:
        jsonl_file = f"batch_request_{int(time.time())}.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for idx, row in data.iterrows():
                filled_prompt = prompt_template.format_messages(**row.to_dict())
                messages = []
                for msg in filled_prompt:
                    msg_dict = msg.model_dump()
                    # Rename "type" to "role" to match API requirements.
                    if "type" in msg_dict:
                        msg_dict["role"] = (
                            "user" if msg_dict["type"] == "human" else msg_dict["type"]
                        )
                        del msg_dict["type"]
                    messages.append(msg_dict)

                # Use the first (and only) value of the row as the custom_id.
                custom_id = row.iloc[0] if pd.notnull(row.iloc[0]) else f"row_{idx}"

                request_object = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": self.COMPLETION_ENDPOINT,
                    "body": {
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "messages": messages,
                    },
                }
                f.write(json.dumps(request_object, ensure_ascii=False) + "\n")
        # Upload the file and then delete it from disk.
        file_id = self._upload_file(jsonl_file)
        os.remove(jsonl_file)
        return file_id

    def _upload_file(self, file_path: str) -> str:
        with open(file_path, "rb") as file_obj:
            response = requests.post(
                self.FILES_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": file_obj},
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
        # Append new batch IDs to the pending batches file.
        with open(self.pending_batches_file, "a", encoding="utf-8") as f:
            f.write(batch_id + "\n")

    def clean_and_parse_json(self, response: str) -> dict:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        response = response.strip("`").strip()
        return json.loads(response)

    def _get_batch_results(self, batch_id: str) -> Optional[pd.DataFrame]:
        """
        Check the status of the given batch_id and, if completed,
        download and parse its JSONL results into a DataFrame.
        """
        status_response = requests.get(
            f"{self.BATCH_API_URL}/{batch_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        status_response.raise_for_status()
        batch_info = status_response.json()
        if batch_info.get("status") != "completed":
            return None

        output_file_id = batch_info.get("output_file_id")
        file_response = requests.get(
            f"{self.FILES_API_URL}/{output_file_id}/content",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        file_response.raise_for_status()
        content = file_response.text

        records = []
        for line in content.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            # Process only if response exists and status code is 200.
            if record.get("response") and record["response"].get("status_code") == 200:
                try:
                    chat_body = record["response"]["body"]
                    choices = chat_body.get("choices")
                    if choices and len(choices) > 0:
                        message_content = choices[0]["message"]["content"]
                        parsed_content = self.clean_and_parse_json(message_content)
                        parsed_content["custom_id"] = record.get("custom_id")
                        records.append(parsed_content)
                except Exception:
                    continue
        if records:
            return pd.DataFrame(records)
        else:
            return pd.DataFrame()

    def retrieve_batch_results(
        self, batch_id: Optional[str] = None, pending_batches_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve batch results either for a singular batch_id or iterate over
        batch ids in a pending_batches_file.
        """
        if batch_id:
            df = self._get_batch_results(batch_id)
            return df if df is not None else pd.DataFrame()
        else:
            file_to_use = pending_batches_file or self.pending_batches_file
            results = []
            pending_ids = []
            with open(file_to_use, "r", encoding="utf-8") as f:
                ids = [line.strip() for line in f.readlines()]
            for bid in ids:
                df_bid = self._get_batch_results(bid)
                if df_bid is None or df_bid.empty:
                    pending_ids.append(bid)
                else:
                    results.append(df_bid)
            # Update the pending file with those still incomplete.
            with open(file_to_use, "w", encoding="utf-8") as f:
                for pid in pending_ids:
                    f.write(pid + "\n")
            if results:
                return pd.concat(results, ignore_index=True)
            else:
                return pd.DataFrame()

    def try_enrich_dataframe(
        self, df: pd.DataFrame, prompt_class: BasePrompt, try_wait: bool = True
    ) -> Optional[pd.DataFrame]:
        prompt_template = prompt_class.get_prompt_template()

        if self.use_batch:
            batch_id = self.generate_batch_openai(df, prompt_template)
            print(f"Batch submitted: {batch_id} (Saved to {self.pending_batches_file})")
            if try_wait:
                # Wait for one minute before attempting to retrieve results.
                time.sleep(60)
                batch_results_df = self.retrieve_batch_results(batch_id=batch_id)
                if batch_results_df.empty:
                    return pd.DataFrame()  # Batch still running or no valid responses.
                else:
                    enriched_df = df.merge(
                        batch_results_df,
                        left_on=df.columns[0],
                        right_on="custom_id",
                        how="left",
                    )
                    # Drop custom_id in the final enriched dataframe.
                    enriched_df = enriched_df.drop(columns=["custom_id"])
                    return enriched_df
            else:
                return pd.DataFrame()
        else:
            synthetic_data = self.generate_batch_local(df, prompt_template)
            parsed_data = [self.clean_and_parse_json(entry) for entry in synthetic_data]
            synthetic_df = pd.DataFrame(parsed_data)
            synthetic_df["Score Explanations"] = synthetic_df[
                "Score Explanations"
            ].apply(lambda d: "; ".join([f"{k}: {v}" for k, v in d.items()]))
            return pd.concat([df, synthetic_df], axis=1)

    def fetch_completed_batches(self) -> List[pd.DataFrame]:
        """
        Check all pending batch IDs from the pending_batches_file. For each completed batch,
        download and format its results into a DataFrame (with the custom_id column preserved).
        Completed batch IDs are removed from pending_batches.txt.
        Returns a list of DataFrames for each completed batch.
        """
        if not os.path.exists(self.pending_batches_file):
            print("No pending batches to fetch.")
            return []
        with open(self.pending_batches_file, "r", encoding="utf-8") as f:
            batch_ids = [line.strip() for line in f.readlines()]
        completed_dfs = []
        still_pending = []
        for batch_id in batch_ids:
            response = requests.get(
                f"{self.BATCH_API_URL}/{batch_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            status = response.json()["status"]
            if status == "completed":
                df_batch = self._get_batch_results(batch_id)
                if df_batch is not None and not df_batch.empty:
                    completed_dfs.append(df_batch)
            elif status in ["failed", "cancelled"]:
                print(f"Batch {batch_id} failed or was cancelled.")
            else:
                still_pending.append(batch_id)
        # Update the pending_batches file with those still incomplete.
        with open(self.pending_batches_file, "w", encoding="utf-8") as f:
            for pid in still_pending:
                f.write(pid + "\n")
        return completed_dfs

    def merge_completed_batches_with_originals(
        self, completed_batches: List[pd.DataFrame], **original_dfs
    ) -> dict:
        """
        For each provided original DataFrame (passed as keyword arguments), this method
        merges relevant completed batch results with the original data. It uses the first column
        of the original DF (assumed to be the unique ID, e.g. 'Artist name' or 'Auction House')
        to match with the 'custom_id' column in the batch results. The merged DataFrame has the
        'custom_id' column dropped.
        Returns a dictionary mapping each key to its merged DataFrame.
        """
        merged_dict = {}
        for key, orig_df in original_dfs.items():
            id_col = orig_df.columns[0]
            # Ensure the original IDs are treated as strings.
            orig_ids = orig_df[id_col].astype(str)
            relevant_batches = []
            for batch_df in completed_batches:
                # Ensure custom_id is string.
                batch_df = batch_df.copy()
                batch_df["custom_id"] = batch_df["custom_id"].astype(str)
                mask = batch_df["custom_id"].isin(orig_ids)
                if not batch_df[mask].empty:
                    relevant_batches.append(batch_df[mask])
            if relevant_batches:
                concatenated = pd.concat(relevant_batches, ignore_index=True)
                merged = orig_df.merge(
                    concatenated, left_on=id_col, right_on="custom_id", how="left"
                )
                merged = merged.drop(columns=["custom_id"])
            else:
                merged = orig_df.copy()
            merged_dict[key] = merged
        return merged_dict


if __name__ == "__main__":
    pending_batch_file_path = f"{ADDITIONAL_DATA_PATH}/pending_batches.txt"
    data_to_enrich = FileManager.read_all_csvs(CSVS_PATH)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    generator = SyntheticDataGenerator(
        model_name="gpt-4o",
        temperature=0.1,
        use_batch=True,
        pending_batches_file=pending_batch_file_path,
    )

    artists = PropertyExtractor.get_unique_dataframe(data_to_enrich, "Artist name")
    # artists_enriched = generator.try_enrich_dataframe(
    #     artists, ArtistPrompt(), try_wait=False
    # )
    # if not artists_enriched.empty:
    #     artists_enriched.to_csv(
    #         f"{ADDITIONAL_DATA_PATH}/artists_{timestamp}.csv", sep=";"
    #     )

    auction_houses = PropertyExtractor.get_unique_dataframe(
        data_to_enrich, "Auction House"
    )
    # auction_houses_enriched = generator.try_enrich_dataframe(
    #     auction_houses, AuctionHousePrompt(), try_wait=False
    # )
    # if not auction_houses_enriched.empty:
    #     auction_houses_enriched.to_csv(
    #         f"{ADDITIONAL_DATA_PATH}/auction_houses_{timestamp}.csv", sep=";"
    #     )

    # # Retrieve completed batches from pending_batches.txt and merge with originals.
    completed_batches_list = generator.fetch_completed_batches()
    if completed_batches_list:
        merged_dict = generator.merge_completed_batches_with_originals(
            completed_batches_list, artists=artists, auction_houses=auction_houses
        )
        for key, df in merged_dict.items():
            if df.shape[1] > 1:  # Exclude if only initial columns are returned earlier
                file_name = f"{ADDITIONAL_DATA_PATH}/merged_{key}_{timestamp}.csv"
                df.to_csv(file_name, sep=";", index=False)
                print(f"Merged results for {key} saved to {file_name}")
            else:
                print(
                    f"Merged results for {key} contain only one column; skipping CSV creation."
                )
    else:
        print("No completed batch results available.")
