import os

def print_django_structure(folder_path, indent=""):
    try:
        entries = sorted(os.listdir(folder_path))
    except FileNotFoundError:
        print(f"❌ Folder not found: {folder_path}")
        return
    except PermissionError:
        print(f"❌ Permission denied: {folder_path}")
        return

    for i, entry in enumerate(entries):
        path = os.path.join(folder_path, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        if os.path.isdir(path):
            print(f"{indent}{connector}📁 {entry}/")
            print_django_structure(path, indent + ("    " if i == len(entries) - 1 else "│   "))
        else:
            icon = "📄"
            if entry == "settings.py":
                icon = "⚙️ "
            elif entry == "urls.py":
                icon = "🌐"
            elif entry == "views.py":
                icon = "👀"
            elif entry == "models.py":
                icon = "🧠"
            elif entry == "admin.py":
                icon = "🛠️"
            elif entry == "apps.py":
                icon = "📦"
            elif entry == "manage.py":
                icon = "🧭"
            elif entry.endswith(".html"):
                icon = "📝"
            print(f"{indent}{connector}{icon} {entry}")

# Example usage
if __name__ == "__main__":
    folder_path = "D:/Coding/0-FYP-WORK-FINAL/Neuro-Insight"

    print(f"\n📂 Django Project Structure for: {folder_path}\n")
    print_django_structure(folder_path)
