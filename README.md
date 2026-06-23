# תמלול עברית מקומי

כלי שורת פקודה לתמלול קובצי MP3, WAV, MP4, OGG ו־M4A בעברית,
באמצעות `faster-whisper` ומודל ברירת המחדל
`ivrit-ai/whisper-large-v3-turbo-ct2`.

## התקנה

נדרשת Python 3.9 ומעלה:

```bash
python -m pip install -r requirements.txt
```

מודל Turbo נשמר בתיקייה `models/ivrit-turbo`. בזמן התמלול הוא נטען
מהדיסק בלבד: אין API, אין מנוי ואין תשלום.

## שימוש

הדפסת התמלול למסך:

```bash
python transcribe.py recording.mp3
```

בזמן התמלול יוצג פס התקדמות שמתעדכן עד 100%. הפס נכתב לערוץ
השגיאות (`stderr`), ולכן אינו מתערבב בטקסט התמלול שמודפס למסך.
אפשר לעצור את טעינת המודל או את התמלול בכל עת באמצעות `Ctrl+C`.
ברירת המחדל מותאמת למהירות על CPU. לחיפוש יסודי ואיטי יותר:

```bash
python transcribe.py recording.mp3 --accurate
```

שמירת קובץ טקסט:

```bash
python transcribe.py recording.mp3 --output transcript.txt
```

יצירת `recording.srt`:

```bash
python transcribe.py recording.mp3 --srt
```

### ציר זמן

הצגת תמלול קריא עם זמן התחלה וסיום לכל מקטע:

```bash
python transcribe.py recording.mp3 --timeline
```

דוגמה:

```text
[00:00:03 - 00:00:08] טקסט התמלול
```

שמירת התמלול עם ציר הזמן לקובץ:

```bash
python transcribe.py recording.mp3 --timeline --output transcript.txt
```

בחירת מודל:

```bash
python transcribe.py recording.mp3 --model small
python transcribe.py recording.mp3 --model medium
python transcribe.py recording.mp3 --model turbo
python transcribe.py recording.mp3 --model large
```

`turbo` הוא מודל ברירת המחדל העברי והמהיר של ivrit.ai. האפשרות `large`
משתמשת במודל העברי הישן והאיטי יותר. האפשרויות `small` ו־`medium`
משתמשות במודלי Whisper הרב־לשוניים המתאימים.

הצגת התקדמות ופרטי שגיאה:

```bash
python transcribe.py recording.mp3 --verbose
```
# Hebrew-Audio-Transcriber
