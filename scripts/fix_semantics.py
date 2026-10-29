"""Hand-reviewed corrections to lines whose automated shortening changed meaning."""
import json
from pathlib import Path
from pipeline.common import save_json

folder = Path(__file__).resolve().parents[1] / "work" / "c__DMV_Clog"
path = folder / "segments.json"
data = json.loads(path.read_text(encoding="utf-8"))
corrections = {
    4: "ಕೆಲಸದ ಬದುಕಿನ ಸವಾಲುಗಳನ್ನು ನಾವೇ ಎದುರಿಸಲು ಸಿದ್ಧರಾಗಬೇಕು.",
    8: "ಸಂಬಂಧದಲ್ಲಿ ತೊಂದರೆ ಬಂದಾಗ ವೇಗವರ್ಧಕ, ಬ್ರೇಕ್ ನಡುವೆ ಗೊಂದಲ ಉಂಟಾಗುತ್ತದೆ; ಅತಿರೇಕವೂ ಇರುತ್ತದೆ.",
    20: "ಇನ್ನೊಬ್ಬರು ನಿರ್ಲಕ್ಷ್ಯ ತೋರುವಂತೆ ಕಾಣುತ್ತಾರೆ.",
    22: "ಮತ್ತೊಬ್ಬರು ಹಸಿವಿನಿಂದ ಕಾಯಬೇಕಾಗುತ್ತದೆ; ಇದು ಇಬ್ಬರಿಗೂ ಅನ್ಯಾಯ.",
    31: "ದೂರವಿರಬೇಕೆಂಬ ಆಸೆಯನ್ನು ನೋವುಂಟುಮಾಡುವಂತೆ ತೋರಿಸಿ, ಮಾಜಿ ಪ್ರೇಮಿಗೆ ಸಂದೇಶ ಕಳುಹಿಸುತ್ತಾರೆ.",
    33: "ಅವರು ಹಾಯಿದೋಣಿ ಅಥವಾ ಮೀನುಗಾರಿಕೆಯಲ್ಲಿ ಮುಳುಗುತ್ತಾರೆ; ಸಮಸ್ಯೆಗಳು ದೇಹದಲ್ಲೂ ಕಾಣಿಸುತ್ತವೆ.",
    37: "ಮಗುವಿಗೆ ಜನ್ಮ ನೀಡಿ, ಬಳಿಕ ಅದರ ಸ್ವಂತ ಆಸೆಗಳಿಗೆ ಅವಕಾಶ ಕೊಡುವುದು ಪೋಷಕರಿಗೆ ದೊಡ್ಡ ಸವಾಲು.",
    38: "ಮಗುವನ್ನು ಕಿರುಕುಳಿಸದೆ, ಅದರದೇ ಭಾವನೆಗಳೊಂದಿಗೆ ಸ್ವತಂತ್ರವಾಗಿ ಬೆಳೆಯಲು ಬಿಡಬೇಕು.",
    51: "ಪ್ರೀತಿಯ ಗಾಡಿ ಹಾದಿ ತಪ್ಪದಂತೆ ಓಡಿಸಲು ಅವರು ಕಲಿಯುತ್ತಾರೆ.",
}
for unit in data["segments"]:
    if unit["id"] in corrections:
        text = corrections[unit["id"]]
        print(unit["id"], unit["kn"], "=>", text)
        unit["kn"] = unit["kn_fit"] = text
        unit["kn_chars"] = len(text)
        (folder / "tts" / f"unit_{unit['id']:04d}.wav").unlink(missing_ok=True)
save_json(path, data)
