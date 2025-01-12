# ADL_Gruppe_3

## Image Classifier (Nico)
- Funktion: classify_image(Mat pImage)
- Input: Bild von opencv
- Output: Klassifikation der Flasche (Bierflasche, Wasser, ...)
## Article Agent (Buket)
- Funktion: generate_article_content(String pClassification)
- Input: String von Flaschenklassifikation
- Output: Image Promt/Image Description zu Flasche, LLM Text zu Flasche
## Article Assembler (Moritz, Dimitri)
- Funktion: assemble_article(String pArticleText, String pImageDescription)
- Input: Beschreibung/Prompt für das Diffusion Model, Text für den Artikel (generiert vom LLM)
- Output: Pfad zum generierten PDF File (mit markdown Template -> Pandoc zur generierung)