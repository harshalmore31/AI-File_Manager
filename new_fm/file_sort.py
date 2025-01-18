import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Generator, Any
from dataclasses import dataclass
from collections import defaultdict
import shutil
import uuid
import datetime
import csv
import hashlib
from rich import print as rprint
from rich.tree import Tree
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import track
from rich.panel import Panel
from rich.console import Console
from rich.style import Style
import yaml
from queue import Queue
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from functools import partial
import signal
import sys

@dataclass
class FileInfo:
    path: Path
    size: int
    category: str
    depth: int
    hash: str = ""
    
    @classmethod
    def from_path(cls, path: Path, base_depth: int) -> Optional['FileInfo']:
        try:
            path = path.resolve()
            if path.is_symlink():
                logging.warning(f"Skipping symbolic link: {path}")
                return None
                
            size = path.stat().st_size
            return cls(
                path=path,
                size=size,
                category='',  # Will be set later
                depth=len(path.parts) - base_depth
            )
        except Exception as e:
            logging.error(f"Failed to process {path}: {e}")
            return None
    
    def calculate_hash(self) -> None:
        """Calculate SHA-256 hash of file content."""
        try:
            sha256_hash = hashlib.sha256()
            with self.path.open('rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(chunk)
            self.hash = sha256_hash.hexdigest()
        except Exception as e:
            logging.error(f"Failed to calculate hash for {self.path}: {e}")
            self.hash = ""

class FileOrganizerConfig:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.default_config = {
            'category_mapping': {
                'documents': ['.pdf', '.docx', '.doc', '.txt', '.pptx', '.xlsx', '.csv'],
                'media': ['.mp4', '.avi', '.mkv', '.jpg', '.jpeg', '.png', '.gif'],
                'archives': ['.zip', '.rar', '.tar', '.gz', '.7z'],
                'code': ['.py', '.js', '.java', '.cpp', '.h', '.css', '.html'],
                'executables': ['.exe', '.msi', '.apk'],
                'others': []
            },
            'min_file_size': 3 * 1024,  # 3KB
            'max_depth': 3,
            'skip_patterns': [
                r'node_modules',
                r'\.git',
                r'\.venv',
                r'__pycache__',
                r'\.idea',
                r'venv',
                r'dist',
                r'build'
            ],
            'duplicate_handling': 'rename',  # Options: rename, skip, overwrite
            'logging': {
                'max_size': 5 * 1024 * 1024,  # 5MB
                'backup_count': 3,
                'level': 'INFO'
            }
        }
        self.config = self.load_config()

    def load_config(self) -> dict:
        """Load or create configuration file."""
        try:
            if not self.config_path.exists():
                self.config_path.write_text(yaml.dump(self.default_config))
                return self.default_config
            
            with self.config_path.open('r') as f:
                config = yaml.safe_load(f)
                
            # Validate and merge with defaults
            return self._validate_config(config)
        except Exception as e:
            logging.error(f"Error loading config: {e}. Using defaults.")
            return self.default_config
    
    def _validate_config(self, config: dict) -> dict:
        """Validate and merge with default config."""
        validated = self.default_config.copy()
        
        for key, value in config.items():
            if key in validated:
                if isinstance(validated[key], dict):
                    validated[key].update(value)
                else:
                    validated[key] = value
        
        return validated

class SmartFileOrganizer:
    def __init__(self, config_path: Path = Path('file_organizer_config.yaml')):
        self.console = Console()
        self.config = FileOrganizerConfig(config_path)
        self.setup_logging()
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.console.print("\n[yellow]Received interrupt signal. Cleaning up...[/]")
        # Implement cleanup logic here if needed
        sys.exit(0)
    
    def setup_logging(self):
        """Setup rotating log handler."""
        log_config = self.config.config['logging']
        handlers = [
            RotatingFileHandler(
                'file_organizer.log',
                maxBytes=log_config['max_size'],
                backupCount=log_config['backup_count']
            ),
            logging.StreamHandler()
        ]
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
    
    def scan_directory(self, root_path: Path) -> Generator[FileInfo, None, None]:
        """Scan directory using generator-based approach."""
        base_depth = len(root_path.resolve().parts)
        
        def should_process(path: Path) -> bool:
            """Check if path should be processed."""
            if path.is_symlink():
                return False
            return not any(re.search(pattern, str(path)) 
                         for pattern in self.config.config['skip_patterns'])
        
        def scan_recursive(current_path: Path) -> Generator[FileInfo, None, None]:
            try:
                if current_path.is_file():
                    file_info = FileInfo.from_path(current_path, base_depth)
                    if (file_info and 
                        file_info.size >= self.config.config['min_file_size']):
                        yield file_info
                elif current_path.is_dir() and should_process(current_path):
                    # Check depth before processing
                    depth = len(current_path.parts) - base_depth
                    if depth <= self.config.config['max_depth']:
                        for path in current_path.iterdir():
                            yield from scan_recursive(path)
            except PermissionError:
                self.console.print(f"[red]Permission denied: {current_path}[/]")
            except Exception as e:
                logging.error(f"Error processing {current_path}: {e}")
        
        yield from scan_recursive(root_path.resolve())
    
    def get_file_category(self, file_info: FileInfo) -> str:
        """Determine file category based on extension."""
        ext = file_info.path.suffix.lower()
        for category, extensions in self.config.config['category_mapping'].items():
            if ext in extensions:
                return category
        return 'others'
    
    def handle_duplicate(self, dest_path: Path) -> Path:
        """Handle duplicate files based on configuration."""
        if not dest_path.exists():
            return dest_path
            
        handling = self.config.config['duplicate_handling']
        if handling == 'skip':
            return None
        elif handling == 'overwrite':
            return dest_path
        else:  # rename
            counter = 1
            while True:
                new_path = dest_path.with_name(
                    f"{dest_path.stem}_{counter}{dest_path.suffix}"
                )
                if not new_path.exists():
                    return new_path
                counter += 1
    
    def move_file(self, file_info: FileInfo, dest_dir: Path) -> bool:
        """Move a single file with verification."""
        try:
            # Calculate source hash if not already done
            if not file_info.hash:
                file_info.calculate_hash()
                
            dest_path = self.handle_duplicate(dest_dir / file_info.path.name)
            if not dest_path:
                logging.info(f"Skipping duplicate file: {file_info.path}")
                return False
                
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(file_info.path), str(dest_path))
            
            # Verify move
            moved_info = FileInfo.from_path(dest_path, len(dest_path.parts))
            if moved_info:
                moved_info.calculate_hash()
                if moved_info.hash != file_info.hash:
                    raise ValueError("File verification failed")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to move {file_info.path}: {e}[/]")
            logging.error(f"Move failed: {e}")
            return False
    
    def organize_files(self, source_dir: Path, operation_type: str = 'category') -> str:
        """Organize files with progress tracking and error handling."""
        operation_id = str(uuid.uuid4())
        organized_dir = source_dir / 'organized_files'
        organized_dir.mkdir(exist_ok=True)
        
        # Collect files first
        files_to_process = list(self.scan_directory(source_dir))
        if not files_to_process:
            self.console.print("[yellow]No files found to organize.[/]")
            return operation_id
        
        # Process files with progress tracking
        with ThreadPoolExecutor() as executor:
            futures = []
            
            for file_info in files_to_process:
                if operation_type == 'extension':
                    category = file_info.path.suffix.lstrip('.') or 'others'
                else:
                    file_info.category = self.get_file_category(file_info)
                    category = file_info.category
                
                dest_dir = organized_dir / category
                futures.append(
                    executor.submit(self.move_file, file_info, dest_dir)
                )
            
            # Track progress
            with self.console.status("[bold green]Processing files...") as status:
                for future in track(as_completed(futures), 
                                 total=len(futures),
                                 description="Moving files"):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Failed to process file: {e}")
        
        return operation_id
    
    def generate_report(self, directory: Path) -> None:
        """Generate detailed analysis report."""
        stats = defaultdict(lambda: {'count': 0, 'size': 0, 'extensions': set()})
        total_size = 0
        
        # Collect statistics
        for file_info in self.scan_directory(directory):
            category = self.get_file_category(file_info)
            stats[category]['count'] += 1
            stats[category]['size'] += file_info.size
            stats[category]['extensions'].add(file_info.path.suffix.lower())
            total_size += file_info.size
        
        # Generate report
        report_path = Path('file_analysis_report.md')
        with report_path.open('w') as f:
            f.write("# File Organization Analysis Report\n\n")
            f.write(f"## Directory: {directory}\n\n")
            
            # Summary table
            table = Table(title="File Statistics")
            table.add_column("Category", style="cyan")
            table.add_column("Count", justify="right", style="magenta")
            table.add_column("Size", justify="right", style="green")
            table.add_column("Extensions", style="blue")
            
            for category, data in sorted(stats.items()):
                table.add_row(
                    category,
                    str(data['count']),
                    self.format_size(data['size']),
                    ", ".join(sorted(data['extensions']))
                )
            
            # Add total
            table.add_row(
                "Total",
                str(sum(d['count'] for d in stats.values())),
                self.format_size(total_size),
                ""
            )
            
            self.console.print(table)
            f.write("\n```\n")
            self.console.print(table, file=f)
            f.write("```\n")
    
    @staticmethod
    def format_size(size: int) -> str:
        """Format size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

def main():
    """Main function with improved error handling and user interaction."""
    organizer = SmartFileOrganizer()
    console = Console()
    
    console.rule("[bold blue]Smart File Organizer[/]")
    
    try:
        # Get directory path with validation
        while True:
            dir_path = Path(Prompt.ask("Enter directory path", default="."))
            try:
                dir_path = dir_path.resolve()
                if dir_path.exists() and dir_path.is_dir():
                    break
                console.print("[red]Invalid directory path. Please try again.[/]")
            except Exception as e:
                console.print(f"[red]Error processing path: {e}[/]")
        
        # Main loop
        while True:
            console.print("\n[bold blue]Choose an operation:[/]")
            console.print("1. Organize by category")
            console.print("2. Organize by extension")
            console.print("3. Generate report")
            console.print("4. Exit")
            
            choice = Prompt.ask(
                "Enter choice",
                choices=['1', '2', '3', '4'],
                default='4'
            )
            
            if choice == '1':
                if Confirm.ask("Organize files by category?"):
                    op_id = organizer.organize_files(dir_path, 'category')
                    console.print(f"[green]Operation complete. ID: {op_id}[/]")
            elif choice == '2':
                if Confirm.ask("Organize files by extension?"):
                    op_id = organizer.organize_files(dir_path, 'extension')
                    console.print(f"[green]Operation complete. ID: {op_id}[/]")
            elif choice == '3':
                organizer.generate_report(dir_path)
                console.print("[green]Report generated successfully.[/]")
            elif choice == '4':
                console.print("[blue]Exiting the program. Goodbye![/]")
                break
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/]")
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()