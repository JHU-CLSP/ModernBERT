import os
from pathlib import Path
from typing import Optional
from composer.core import Callback, State, Event
from composer.loggers import Logger
from composer.utils import dist
from huggingface_hub import HfApi
import logging

log = logging.getLogger(__name__)

class HuggingFaceSync(Callback):
    """
    Simple callback that syncs model checkpoints to a HuggingFace repository.
    Will block training during upload to ensure consistency.
    """
    
    def __init__(
        self,
        repo_id: str,
        model_save_folder: str,
        token: Optional[str] = None,
        repo_can_exist: bool = False,
    ):
        print(f"Initializing HuggingFaceSync with repo_id: {repo_id}, model_save_folder: {model_save_folder}")
        self.repo_id = repo_id
        self.model_save_folder = Path(model_save_folder)
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError("HuggingFace token must be provided either through init or HF_TOKEN env var")
            
        self.api = HfApi(token=self.token)
        self.last_checkpoint = None
        
        # Initialize repo if needed
        try:
            self.api.repo_info(repo_id, repo_type="model")
            log.info(f"Connected to existing repo: {repo_id}")
        except Exception:
            if not repo_can_exist:
                raise ValueError(f"Repo {repo_id} does not exist and repo_can_exist is False")
                
            
        # Get initial state of repo
        self.uploaded_files = set(self.api.list_repo_files(repo_id, repo_type="model"))
        log.info(f"Found {len(self.uploaded_files)} existing files in repo")

    def _get_new_checkpoint_files(self) -> set:
        """Get list of new checkpoint files that haven't been uploaded."""
        local_files = []
        for path in self.model_save_folder.rglob('*'):
            if path.is_file() and path.name.endswith("-rank0.pt"):
                relative_path = str(path.relative_to(self.model_save_folder))
                if "/" not in relative_path:  # Only get files in main folder
                    local_files.append(relative_path)
                    
        new_files = set(local_files) - self.uploaded_files
        return new_files

    def _sync_to_hub(self):
        """Sync new checkpoint files to HuggingFace. Only runs on rank 0."""
        # Skip if not on rank 0
        if dist.get_global_rank() != 0:
            return
        new_files = self._get_new_checkpoint_files()
        
        if not new_files:
            log.info("No new files to upload")
            return
            
        log.info(f"Found {len(new_files)} new files to upload")
        
        # Upload each file synchronously
        for file in sorted(new_files):
            try:
                local_path = self.model_save_folder / file
                log.info(f"Uploading: {file}")
                
                self.api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=file,
                    repo_id=self.repo_id,
                    repo_type="model"
                )
                
                self.uploaded_files.add(file)
                log.info(f"Successfully uploaded: {file}")
                
            except Exception as e:
                log.error(f"Error uploading {file}: {str(e)}", exc_info=True)
                raise  # Re-raise to stop training if upload fails

    def batch_checkpoint(self, state: State, logger: Logger) -> None:
        """Check for new checkpoints after batch checkpoint."""
        current_checkpoint = state.timestamp.batch.value
        if current_checkpoint != self.last_checkpoint:
            log.info(f"New checkpoint detected: {current_checkpoint}")
            self._sync_to_hub()
            self.last_checkpoint = current_checkpoint

    def iteration_checkpoint(self, state: State, logger: Logger) -> None:
        """Check for new checkpoints after iteration checkpoint."""
        current_checkpoint = state.timestamp.batch.value
        if current_checkpoint != self.last_checkpoint:
            log.info(f"New checkpoint detected: {current_checkpoint}")
            self._sync_to_hub()
            self.last_checkpoint = current_checkpoint

    def epoch_checkpoint(self, state: State, logger: Logger) -> None:
        """Check for new checkpoints after epoch checkpoint."""
        current_checkpoint = state.timestamp.batch.value
        if current_checkpoint != self.last_checkpoint:
            log.info(f"New checkpoint detected: {current_checkpoint}")
            self._sync_to_hub()
            self.last_checkpoint = current_checkpoint