import re
import os
import sys

def increment_version(version_str):
    """Increments the patch version of a semver string."""
    parts = version_str.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid version string: {version_str}")
    major, minor, patch = map(int, parts)
    patch += 1
    return f"{major}.{minor}.{patch}"

def update_file(filepath, pattern, replacement_format, new_version, optional=False):
    """Updates the version in a file using a regex pattern."""
    if not os.path.exists(filepath):
        if optional:
            print(f"Optional file not found: {filepath}. Skipping.")
            return True
        print(f"Error: Required file not found: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Use \g<1> for group 1 to avoid ambiguity with version numbers
    replacement = replacement_format.format(new_version).replace(r'\1', r'\g<1>')
    new_content = re.sub(pattern, replacement, content)

    if new_content == content:
        if optional:
            print(f"No changes made to optional file {filepath}. Pattern might not have matched. Skipping.")
            return True
        print(f"Error: No changes made to required file {filepath}. Pattern might not have matched.")
        return False

    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Updated {filepath}")
    return True

def main():
    # 1. Get current version from pyproject.toml
    pyproject_path = 'pyproject.toml'
    if not os.path.exists(pyproject_path):
        print(f"Error: {pyproject_path} not found.")
        sys.exit(1)

    with open(pyproject_path, 'r') as f:
        content = f.read()

    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)

    current_version = match.group(1)
    new_version = increment_version(current_version)
    print(f"Incrementing version: {current_version} -> {new_version}")

    # 2. Update files
    success = True

    # Update pyproject.toml (Required)
    if not update_file(pyproject_path, r'(version\s*=\s*)"[^"]+"', r'\1"{}"', new_version):
        success = False

    # Update redis_robot_comm/__init__.py (Required)
    if not update_file('redis_robot_comm/__init__.py', r'(__version__\s*=\s*)"[^"]+"', r'\1"{}"', new_version):
        success = False

    # Update docs/api.md (Optional)
    update_file('docs/api.md', r'(- \*\*Package Version\*\*:\s*)\d+\.\d+\.\d+', r'\1{}', new_version, optional=True)

    if not success:
        print("Required files failed to update.")
        sys.exit(1)

    # 3. Output for GitHub Actions
    github_output = os.getenv('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"new_version={new_version}\n")

    print(f"Successfully updated version to {new_version}")

if __name__ == "__main__":
    main()
