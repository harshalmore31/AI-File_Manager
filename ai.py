import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import json
from openai import OpenAI
import time
from datetime import datetime
import logging
from collections import defaultdict
import humanize
from jinja2 import Template
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# System instructions for the LLM
SYSTEM_INSTRUCTIONS = """You are a File Manager Bot specialized in organizing files and directories efficiently. Your task is to:

1. Analyze the given file hierarchy structure
2. Identify patterns in:
   - File names
   - Extensions
   - Content categories
   - Date patterns
   - Version patterns
   - Project relationships

3. Propose a new, optimized hierarchy that:
   - Groups related files logically
   - Separates different types of content
   - Maintains clear naming conventions
   - Preserves version history
   - Improves searchability and accessibility

4. Output Format Requirements:
   - Provide the complete new directory structure in JSON format
   - List specific move/copy operations needed
   - Explain the rationale for major organizational changes

Example Input Structure:
{
    "type": "directory",
    "name": "project_root",
    "contents": [
        {
            "type": "file",
            "name": "doc1_v2.pdf",
            "extension": ".pdf"
        },
        {
            "type": "directory",
            "name": "random_folder",
            "contents": [
                {
                    "type": "file",
                    "name": "doc1_v1.pdf",
                    "extension": ".pdf"
                }
            ]
        }
    ]
}

Example Output Structure:
{
    "type": "directory",
    "name": "organized_root",
    "contents": [
        {
            "type": "directory",
            "name": "documents",
            "contents": [
                {
                    "type": "directory",
                    "name": "doc1",
                    "contents": [
                        {
                            "type": "file",
                            "name": "doc1_v1.pdf",
                            "source": "project_root/random_folder/doc1_v1.pdf"
                        },
                        {
                            "type": "file",
                            "name": "doc1_v2.pdf",
                            "source": "project_root/doc1_v2.pdf"
                        }
                    ]
                }
            ]
        }
    ]
}"""

class LLMClient:
    def __init__(self, api_key: str = None):
        """Initialize the LLM client with API key."""
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key or os.getenv("NVIDIA_API_KEY")
        )
    
    def _structure_to_text(self, structure: Dict) -> str:
        """Convert file structure to concise text description."""
        def get_file_types(struct):
            extensions = defaultdict(int)
            total_files = 0
            for item in struct["contents"]:
                if item["type"] == "file":
                    ext = item["extension"]
                    extensions[ext] += 1
                    total_files += 1
                elif item["type"] == "directory":
                    sub_ext, sub_total = get_file_types(item)
                    for ext, count in sub_ext.items():
                        extensions[ext] += count
                    total_files += sub_total
            return extensions, total_files

        extensions, total_files = get_file_types(structure)
        
        # Create concise summary
        summary = f"Directory contains {total_files} files larger than 3MB:\n"
        for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
            summary += f"- {count} {ext} files\n"
        
        return summary

    def get_organization_proposal(self, current_structure: Dict) -> Dict:
        """Get organization proposal from LLM."""
        # Convert structure to text summary
        structure_summary = self._structure_to_text(current_structure)
        
        prompt = f"""Please organize these files into a logical structure:
{structure_summary}
Respond with a JSON structure showing the proposed organization. Group similar file types and create meaningful categories."""

        try:
            completion = self.client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                top_p=1,
                max_tokens=2048,
                stream=True
            )
            
            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    response_text += chunk.choices[0].delta.content
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting LLM proposal: {e}")
            return None

class FileSystemScanner:
    def __init__(self, root_directory: str):
        """Initialize scanner with root directory."""
        self.root_directory = Path(root_directory)
        self.file_structure = {}
        self.ignored_patterns = {
            '.git', '__pycache__', 'node_modules', '.env', 'temp', 'tmp',
            '.vscode', '.idea', 'build', 'dist', 'bin', 'obj'
        }
        self.ignored_extensions = {
            '.dll', '.sys', '.exe', '.ini', '.cfg', '.log', '.tmp', '.temp',
            '.cache', '.manifest', '.pdb', '.msi', '.dat', '.bin'
        }
        self.MIN_FILE_SIZE = 3 * 1024 * 1024  # 3MB in bytes
    
    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        return (any(pattern in str(path) for pattern in self.ignored_patterns) or
                path.suffix.lower() in self.ignored_extensions)
        
    def scan_directory(self) -> Dict:
        """Scans the directory and creates a hierarchical structure."""
        logger.info(f"Starting directory scan at: {self.root_directory}")
        
        def scan_recursive(current_path: Path) -> Dict:
            if self.should_ignore(current_path):
                return None
                
            structure = {
                "type": "directory",
                "name": current_path.name,
                "path": str(current_path),  # Add full path
                "contents": []
            }
            
            try:
                for item in current_path.iterdir():
                    if item.is_file():
                        # Skip files smaller than MIN_FILE_SIZE and system files
                        if (item.stat().st_size >= self.MIN_FILE_SIZE and 
                            not self.should_ignore(item)):
                            file_info = {
                                "type": "file",
                                "name": item.name,
                                "path": str(item),  # Add full path
                                "size": item.stat().st_size,
                                "extension": item.suffix.lower()
                            }
                            structure["contents"].append(file_info)
                    elif item.is_dir() and not self.should_ignore(item):
                        subdir_structure = scan_recursive(item)
                        if subdir_structure and subdir_structure["contents"]:  # Only add non-empty directories
                            structure["contents"].append(subdir_structure)
            except PermissionError:
                logger.warning(f"Permission denied: {current_path}")
            except Exception as e:
                logger.error(f"Error scanning {current_path}: {e}")
                
            return structure if structure["contents"] else None
        
        self.file_structure = scan_recursive(self.root_directory)
        if not self.file_structure:
            # Create empty root structure if no files found
            self.file_structure = {
                "type": "directory",
                "name": self.root_directory.name,
                "path": str(self.root_directory),
                "contents": []
            }
        logger.info("Directory scan completed")
        return self.file_structure

class AIFileOrganizer:
    def __init__(self, file_structure: Dict, llm_client: LLMClient):
        """Initialize organizer with file structure and LLM client."""
        self.file_structure = file_structure
        self.llm_client = llm_client
        self.proposed_structure = {}
        self.file_patterns = self._extract_patterns()
        
    def _extract_patterns(self) -> Dict:
        """Extracts patterns from file names and extensions."""
        patterns = {
            "extensions": {},
            "prefixes": {},
            "date_patterns": [],
            "version_patterns": [],
            "size_categories": {
                "small": [],
                "medium": [],
                "large": []
            }
        }
        
        def analyze_recursive(structure: Dict):
            for item in structure["contents"]:
                if item["type"] == "file":
                    # Analyze extensions
                    ext = item["extension"]
                    patterns["extensions"][ext] = patterns["extensions"].get(ext, 0) + 1
                    
                    # Analyze name patterns
                    name_without_ext = Path(item["name"]).stem
                    prefix = name_without_ext.split('_')[0]
                    patterns["prefixes"][prefix] = patterns["prefixes"].get(prefix, 0) + 1
                    
                    # Analyze file sizes
                    size = item["size"]
                    if size < 1024 * 1024:  # < 1MB
                        patterns["size_categories"]["small"].append(item["path"])
                    elif size < 100 * 1024 * 1024:  # < 100MB
                        patterns["size_categories"]["medium"].append(item["path"])
                    else:
                        patterns["size_categories"]["large"].append(item["path"])
                    
                elif item["type"] == "directory":
                    analyze_recursive(item)
        
        analyze_recursive(self.file_structure)
        return patterns
        
    def analyze_structure(self) -> Dict:
        """Analyzes the file structure and generates a proposed organization."""
        logger.info("Starting structure analysis")
        
        # Get proposal from LLM
        self.proposed_structure = self.llm_client.get_organization_proposal(self.file_structure)
        
        if not self.proposed_structure:
            logger.warning("LLM proposal failed, falling back to basic organization")
            return self._generate_basic_structure()
        
        logger.info("Structure analysis completed")
        return self.proposed_structure
    
    def _generate_basic_structure(self) -> Dict:
        """Fallback method for basic file organization."""
        basic_structure = {
            "type": "directory",
            "name": "organized_files",
            "contents": []
        }
        
        # Create categories based on common file types
        categories = {
            "documents": [".pdf", ".doc", ".docx", ".txt", ".xlsx", ".ppt", ".pptx"],
            "media": [".jpg", ".jpeg", ".png", ".gif", ".mp4", ".avi", ".mov"],
            "archives": [".zip", ".rar", ".7z", ".tar", ".gz"],
            "code": [".py", ".js", ".java", ".cpp", ".html", ".css"]
        }
        
        for category, extensions in categories.items():
            category_files = []
            for item in self._get_all_files(self.file_structure):
                if item["extension"] in extensions:
                    category_files.append(item)
            
            if category_files:  # Only add category if it has files
                basic_structure["contents"].append({
                    "type": "directory",
                    "name": category,
                    "contents": category_files
                })
        
        return basic_structure
    
    def _get_all_files(self, structure: Dict) -> List[Dict]:
        """Helper method to get all files from structure."""
        files = []
        for item in structure.get("contents", []):
            if item["type"] == "file":
                files.append(item)
            elif item["type"] == "directory":
                files.extend(self._get_all_files(item))
        return files
    
class FileSystemReorganizer:
    def __init__(self, original_structure: Dict, proposed_structure: Dict):
        """Initialize reorganizer with original and proposed structures."""
        self.original_structure = original_structure
        self.proposed_structure = proposed_structure
        self.operations = []
        self.executed_operations = []
    
    def _determine_target_location(self, file_item: Dict, target_structure: Dict, target_path: Path) -> Optional[Path]:
        """Determines the target location for a file based on the proposed structure."""
        file_name = file_item["name"]
        file_ext = file_item["extension"]
        
        # Look for an appropriate directory in the target structure
        for item in target_structure.get("contents", []):
            if item["type"] == "directory":
                # Check if this directory is meant for this type of file
                if (file_ext in item["name"].lower() or  # Extension-based directory
                    any(category in item["name"].lower() for category in ["documents", "media", "archives", "code"])):
                    return target_path / item["name"] / file_name
        
        # If no specific directory found, place in target root
        return target_path / file_name
    
    def _find_matching_target_dir(self, source_dir: Dict, target_contents: List[Dict]) -> Optional[Dict]:
        """Finds matching target directory for source directory."""
        source_name = source_dir["name"].lower()
        
        for target_item in target_contents:
            if (target_item["type"] == "directory" and 
                (target_item["name"].lower() == source_name or
                 target_item["name"].lower() in source_name or
                 source_name in target_item["name"].lower())):
                return target_item
        
        return None
    
    def plan_reorganization(self) -> List[str]:
        """Plans the reorganization and returns list of operations."""
        logger.info("Planning reorganization")
        self.operations = []
        
        # Start with creating the root directory of proposed structure
        root_path = Path(self.original_structure["path"]).parent / "organized_files"
        self.operations.append(f"CREATE_DIR: {root_path}")
        
        def plan_recursive(current: Dict, target: Dict, current_path: Path, target_path: Path):
            # Create target directory if it doesn't exist
            if target["type"] == "directory":
                self.operations.append(f"CREATE_DIR: {target_path}")
            
            # Process all items in current structure
            for item in current.get("contents", []):
                if item["type"] == "file":
                    # Determine where this file should go in new structure
                    new_location = self._determine_target_location(item, target, target_path)
                    if new_location:
                        self.operations.append(f"MOVE: {item['path']} → {new_location}")
                
                elif item["type"] == "directory":
                    # Find matching directory in target structure
                    matching_target = self._find_matching_target_dir(item, target.get("contents", []))
                    if matching_target:
                        new_target_path = target_path / matching_target["name"]
                        plan_recursive(item, matching_target, 
                                    Path(item["path"]),
                                    new_target_path)
        
        # Start planning from root
        plan_recursive(
            self.original_structure,
            self.proposed_structure,
            Path(self.original_structure["path"]),
            root_path
        )
        
        logger.info(f"Planned {len(self.operations)} operations")
        return self.operations
    
    def execute_reorganization(self, dry_run: bool = True) -> List[str]:
        """Executes the reorganization based on planned operations."""
        if not self.operations:
            self.plan_reorganization()
        
        if dry_run:
            return self.operations
        
        logger.info("Executing reorganization")
        self.executed_operations = []
        
        try:
            for operation in self.operations:
                op_type, paths = operation.split(": ", 1)
                
                if op_type == "CREATE_DIR":
                    path = Path(paths)
                    path.mkdir(parents=True, exist_ok=True)
                    self.executed_operations.append(f"Created directory: {path}")
                
                elif op_type == "MOVE":
                    source, target = paths.split(" → ")
                    source_path = Path(source)
                    target_path = Path(target)
                    
                    if source_path.exists():
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(source_path), str(target_path))
                        self.executed_operations.append(f"Moved: {source} → {target}")
                    else:
                        logger.warning(f"Source file not found: {source}")
                
        except Exception as e:
            logger.error(f"Error during reorganization: {e}")
            raise
        
        logger.info("Reorganization completed successfully")
        return self.executed_operations

def main():
    """Main function to run the file organization system."""
    try:
        # Get user inputs
        root_dir = input("Enter the directory path to organize: ")
        api_key = input("Enter your NVIDIA API key (press Enter to use environment variable): ") or None
        
        # Initialize components
        scanner = FileSystemScanner(root_dir)
        llm_client = LLMClient(api_key)
        
        # Scan directory
        print("\nScanning directory structure...")
        current_structure = scanner.scan_directory()
        
        # Analyze and get proposal
        print("\nAnalyzing files and generating organization proposal...")
        organizer = AIFileOrganizer(current_structure, llm_client)
        proposed_structure = organizer.analyze_structure()
        
        if not proposed_structure:
            print("\nFalling back to basic organization structure...")
            proposed_structure = organizer._generate_basic_structure()
        
        # Generate report
        print("\nGenerating comprehensive report...")
        report_generator = ReportGenerator(current_structure, proposed_structure, organizer.file_patterns)
        report = report_generator.generate_report()
        
        print(f"\nReport generated successfully! Check the 'report' directory for details.")
        
        # Ask for confirmation
        print("\nPlease review the report in the 'report' directory.")
        if input("\nDo you want to proceed with the file reorganization? (y/n): ").lower() == 'y':
            try:
                reorganizer = FileSystemReorganizer(current_structure, proposed_structure)
                operations = reorganizer.plan_reorganization()
                
                print("\nProposed Operations:")
                for operation in operations:
                    print(f"- {operation}")
                
                if input("\nConfirm execution of these operations? (y/n): ").lower() == 'y':
                    print("\nExecuting reorganization...")
                    executed_ops = reorganizer.execute_reorganization(dry_run=False)
                    print("\nExecuted Operations:")
                    for op in executed_ops:
                        print(f"- {op}")
                    print("\nReorganization completed successfully!")
                else:
                    print("\nReorganization cancelled.")
            except Exception as e:
                logger.error(f"Error during reorganization: {e}")
                print(f"\nError during reorganization: {e}")
                return 1
        else:
            print("\nProcess cancelled.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\nError: {e}")
        return 1
    
    return 0

# Additional imports for report generation

class ReportGenerator:
    def __init__(self, current_structure: Dict, proposed_structure: Dict, file_patterns: Dict):
        """Initialize report generator with structures and patterns."""
        self.current_structure = current_structure
        self.proposed_structure = proposed_structure
        self.file_patterns = file_patterns
        self.report_dir = Path("report")
        self.report_dir.mkdir(exist_ok=True)
        
    def generate_tree_structure(self, structure: Dict, prefix="", is_last=True) -> str:
        """Generate tree-like structure representation."""
        tree = ""
        connector = "└── " if is_last else "├── "
        tree += prefix + connector + structure["name"] + "\n"
        
        if structure["type"] == "directory":
            new_prefix = prefix + ("    " if is_last else "│   ")
            contents = structure.get("contents", [])
            
            for i, item in enumerate(contents):
                tree += self.generate_tree_structure(
                    item,
                    new_prefix,
                    i == len(contents) - 1
                )
        
        return tree
    
    def create_size_distribution_chart(self):
        """Create and save size distribution chart."""
        sizes = defaultdict(int)
        
        def collect_sizes(structure):
            if structure["type"] == "file":
                size = structure.get("size", 0)
                if size < 1024 * 1024:  # < 1MB
                    sizes["< 1MB"] += 1
                elif size < 10 * 1024 * 1024:  # < 10MB
                    sizes["1-10MB"] += 1
                elif size < 100 * 1024 * 1024:  # < 100MB
                    sizes["10-100MB"] += 1
                else:
                    sizes["> 100MB"] += 1
            elif structure["type"] == "directory":
                for item in structure.get("contents", []):
                    collect_sizes(item)
        
        collect_sizes(self.current_structure)
        
        plt.figure(figsize=(10, 6))
        plt.bar(sizes.keys(), sizes.values())
        plt.title("File Size Distribution")
        plt.ylabel("Number of Files")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.report_dir / "size_distribution.png")
        plt.close()
    
    def generate_report(self) -> str:
        """Generate comprehensive report."""
        # Create size distribution chart
        self.create_size_distribution_chart()
        
        # Generate tree structures
        current_tree = self.generate_tree_structure(self.current_structure)
        proposed_tree = self.generate_tree_structure(self.proposed_structure)
        
        # Calculate statistics
        total_files = 0
        total_size = 0
        extension_stats = defaultdict(int)
        
        def collect_stats(structure):
            nonlocal total_files, total_size
            if structure["type"] == "file":
                total_files += 1
                total_size += structure.get("size", 0)
                extension_stats[structure.get("extension", "no_ext")] += 1
            elif structure["type"] == "directory":
                for item in structure.get("contents", []):
                    collect_stats(item)
        
        collect_stats(self.current_structure)
        
        # Create report template
        template_str = """# File System Analysis Report
    Generated on: {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}

    ## Overview
    - Total Files: {{ total_files }}
    - Total Size: {{ humanize.naturalsize(total_size) }}

    ## File Extensions
    {% for ext, count in extension_stats.items() %}
    - {{ ext }}: {{ count }} files
    {% endfor %}

    ## Current Directory Structure
    ```
    {{ current_tree }}
    ```

    ## Proposed Directory Structure
    ```
    {{ proposed_tree }}
    ```

    ## Size Distribution
    ![Size Distribution](size_distribution.png)

    ## Organization Analysis
    {% for category, files in file_patterns['size_categories'].items() %}
    ### {{ category.title() }} Files:
    {% for file in files[:5] %}
    - {{ Path(file).name }}
    {% endfor %}
    {% if files|length > 5 %}
    ... and {{ files|length - 5 }} more
    {% endif %}
    {% endfor %}
    """
        
        # Render template
        template = Template(template_str)
        report = template.render(
            datetime=datetime,
            humanize=humanize,
            total_files=total_files,
            total_size=total_size,
            extension_stats=dict(sorted(extension_stats.items())),
            current_tree=current_tree,
            proposed_tree=proposed_tree,
            file_patterns=self.file_patterns,
            Path=Path
        )
        
        # Save report with UTF-8 encoding
        report_path = self.report_dir / "analysis_report.md"
        try:
            with open(report_path, "w", encoding='utf-8') as f:
                f.write(report)
        except Exception as e:
            logger.error(f"Error writing report: {e}")
            # Fallback to ASCII-only output if UTF-8 fails
            report = report.encode('ascii', 'replace').decode('ascii')
            with open(report_path, "w") as f:
                f.write(report)
        
        return report

def main():
    """Main function to run the file organization system."""
    try:
        # Get user inputs
        root_dir = input("Enter the directory path to organize: ")
        api_key = input("Enter your NVIDIA API key (press Enter to use environment variable): ") or None
        
        # Initialize components
        scanner = FileSystemScanner(root_dir)
        llm_client = LLMClient(api_key)
        
        # Scan directory
        print("\nScanning directory structure...")
        current_structure = scanner.scan_directory()
        
        # Analyze and get proposal
        print("\nAnalyzing files and generating organization proposal...")
        organizer = AIFileOrganizer(current_structure, llm_client)
        proposed_structure = organizer.analyze_structure()
        
        # Generate report
        print("\nGenerating comprehensive report...")
        report_generator = ReportGenerator(current_structure, proposed_structure, organizer.file_patterns)
        report = report_generator.generate_report()
        
        print(f"\nReport generated successfully! Check the 'report' directory for details.")
        
        # Ask for confirmation
        print("\nPlease review the report in the 'report' directory.")
        if input("\nDo you want to proceed with the file reorganization? (y/n): ").lower() == 'y':
            reorganizer = FileSystemReorganizer(current_structure, proposed_structure)
            operations = reorganizer.plan_reorganization()
            
            print("\nProposed Operations:")
            for operation in operations:
                print(f"- {operation}")
            
            if input("\nConfirm execution of these operations? (y/n): ").lower() == 'y':
                print("\nExecuting reorganization...")
                executed_ops = reorganizer.execute_reorganization(dry_run=False)
                print("\nExecuted Operations:")
                for op in executed_ops:
                    print(f"- {op}")
                print("\nReorganization completed successfully!")
            else:
                print("\nReorganization cancelled.")
        else:
            print("\nProcess cancelled.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\nError: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)