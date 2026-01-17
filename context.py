#!/usr/bin/env python3
"""
Context Generator: Recursively concatenates directory contents into a single text file.

Takes a directory path and creates a .txt file containing all files with section headers
preserving original file names and relative paths. Respects .gitignore rules.
"""

import os
import sys
from pathlib import Path
from typing import Set, Optional
import argparse

try:
    import gitignore_parser
    HAS_GITIGNORE_PARSER = True
except ImportError:
    HAS_GITIGNORE_PARSER = False
    print("Warning: gitignore_parser not installed. Install with: pip install gitignore-parser")
    print("Will use basic .gitignore parsing instead.")


def parse_gitignore(gitignore_path: Path) -> Optional[object]:
    """Parse .gitignore file and return a matcher function."""
    if not gitignore_path.exists():
        return None
    
    if HAS_GITIGNORE_PARSER:
        return gitignore_parser.parse_gitignore_file(gitignore_path)
    else:
        # Basic gitignore parsing (simplified)
        patterns = []
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
        return patterns


def should_ignore(path: Path, gitignore_matcher, root: Path) -> bool:
    """Check if a path should be ignored based on .gitignore rules."""
    if gitignore_matcher is None:
        return False
    
    if HAS_GITIGNORE_PARSER:
        # gitignore_parser expects relative path from gitignore file location
        try:
            rel_path = path.relative_to(root)
            return gitignore_matcher(rel_path)
        except ValueError:
            return False
    else:
        # Basic pattern matching
        rel_path = str(path.relative_to(root))
        for pattern in gitignore_matcher:
            # Simple pattern matching (basic implementation)
            if pattern in rel_path or rel_path.endswith(pattern):
                return True
        return False


def is_binary_file(file_path: Path) -> bool:
    """Check if a file is likely binary."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(8192)
            # Check for null bytes (common in binary files)
            if b'\x00' in chunk:
                return True
            # Check if file contains mostly printable text
            try:
                chunk.decode('utf-8')
            except UnicodeDecodeError:
                return True
    except Exception:
        return True
    return False


def get_file_content(file_path: Path) -> str:
    """Read file content, handling encoding issues."""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return f"[Error reading file: {e}]"
    return "[Error: Could not decode file with any encoding]"


def find_gitignore(root: Path) -> Optional[Path]:
    """Find .gitignore file in the root directory."""
    gitignore_path = root / '.gitignore'
    if gitignore_path.exists():
        return gitignore_path
    return None


def collect_files(directory: Path, root: Path, gitignore_matcher, 
                 collected: Set[Path], gitignore_path: Optional[Path] = None) -> None:
    """Recursively collect files from directory, respecting .gitignore."""
    try:
        for item in directory.iterdir():
            # Skip hidden files/directories (except .gitignore itself)
            if item.name.startswith('.') and item.name != '.gitignore':
                # Check if we should process .gitignore files in subdirectories
                if item.name == '.gitignore':
                    # Parse this .gitignore for this subdirectory
                    sub_gitignore = parse_gitignore(item)
                    if sub_gitignore:
                        # Update matcher to consider this subdirectory's gitignore
                        # For simplicity, we'll use the root gitignore
                        pass
                continue
            
            # Check if item should be ignored
            if should_ignore(item, gitignore_matcher, root):
                continue
            
            if item.is_file():
                # Skip binary files
                if is_binary_file(item):
                    continue
                collected.add(item)
            elif item.is_dir():
                # Skip .llm-context directory to avoid recursion
                if item.name == '.llm-context':
                    continue
                # Recursively process subdirectory
                collect_files(item, root, gitignore_matcher, collected, gitignore_path)
    except PermissionError:
        print(f"Warning: Permission denied accessing {directory}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error processing {directory}: {e}", file=sys.stderr)


def generate_context(directory: Path, output_dir: Path) -> Path:
    """Generate context file from directory."""
    directory = directory.resolve()
    output_dir = output_dir.resolve()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and parse .gitignore
    gitignore_path = find_gitignore(directory)
    gitignore_matcher = None
    if gitignore_path:
        gitignore_matcher = parse_gitignore(gitignore_path)
    
    # Collect all files
    print(f"Scanning directory: {directory}")
    collected_files = set()
    collect_files(directory, directory, gitignore_matcher, collected_files, gitignore_path)
    
    # Sort files by path for consistent output
    sorted_files = sorted(collected_files, key=lambda p: str(p))
    
    print(f"Found {len(sorted_files)} files to include")
    
    # Generate output filename
    dir_name = directory.name or "root"
    output_file = output_dir / f"{dir_name}_context.txt"
    
    # Write concatenated content
    print(f"Writing context file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as out:
        for file_path in sorted_files:
            # Calculate relative path from root directory
            try:
                rel_path = file_path.relative_to(directory)
            except ValueError:
                rel_path = file_path
            
            # Write section header
            out.write(f"\n{'='*80}\n")
            out.write(f"FILE: {rel_path}\n")
            out.write(f"PATH: {file_path}\n")
            out.write(f"{'='*80}\n\n")
            
            # Write file content
            try:
                content = get_file_content(file_path)
                out.write(content)
                if not content.endswith('\n'):
                    out.write('\n')
            except Exception as e:
                out.write(f"[Error reading file: {e}]\n")
    
    print(f"Context file generated: {output_file}")
    print(f"Total files included: {len(sorted_files)}")
    
    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate a context file from a directory, respecting .gitignore rules'
    )
    parser.add_argument(
        'directory',
        type=str,
        nargs='?',
        default='.',
        help='Directory to process (default: current directory)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: .llm-context in script location)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    input_dir = Path(args.directory).resolve()
    
    if not input_dir.exists():
        print(f"Error: Directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = script_dir / '.llm-context'
    
    # Generate context
    try:
        output_file = generate_context(input_dir, output_dir)
        print(f"\nSuccess! Context file created at: {output_file}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
