import kagglehub

# Download latest version
path = kagglehub.competition_download('bt-4012-competition-2026')

print("Path to competition files:", path)