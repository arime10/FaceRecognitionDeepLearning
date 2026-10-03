import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import win32com.client

import config

def set_cell_background(cell, fill_hex):
    """Hücre arka plan rengini ayarlar."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Hücre içi padding ayarlar."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_academic_docx(output_docx_path):
    print(">> Word belgesi (.docx) oluşturuluyor...")
    doc = docx.Document()

    # Sayfa Kenar Boşlukları (Standart Akademik 2.54 cm)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # 1. BAŞLIK
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("Biyometrik Kimlik Doğrulama Sistemlerinde Derin Evrişimli Sinir Ağları ve Açıklanabilir Yapay Zeka (Grad-CAM) Tabanlı Sahte Yüz Tespiti")
    title_run.font.name = "Calibri"
    title_run.font.size = Pt(18)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(15, 23, 42) # Deep Slate

    # İngilizce Başlık
    en_title_p = doc.add_paragraph()
    en_title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    en_title_run = en_title_p.add_run("Deep Convolutional Neural Networks and Explainable AI (Grad-CAM) Based Face Anti-Spoofing Detection for Biometric Authentication Systems")
    en_title_run.font.name = "Calibri"
    en_title_run.font.size = Pt(11)
    en_title_run.font.italic = True
    en_title_run.font.color.rgb = RGBColor(100, 116, 139)

    # 2. YAZAR BİLGİSİ
    author_p = doc.add_paragraph()
    author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_p.paragraph_format.space_after = Pt(18)
    
    author_run = author_p.add_run("Emirhan Dağ\n")
    author_run.font.name = "Calibri"
    author_run.font.size = Pt(13)
    author_run.font.bold = True
    author_run.font.color.rgb = RGBColor(37, 99, 235) # Blue

    affil_run = author_p.add_run("Biyometrik Güvenlik ve Sahte Yüz Tespit Sistemi Proje Raporu - 2026")
    affil_run.font.name = "Calibri"
    affil_run.font.size = Pt(10)
    affil_run.font.color.rgb = RGBColor(71, 85, 105)

    # 3. ÖZET (ABSTRACT KUTUSU)
    abstract_table = doc.add_table(rows=1, cols=1)
    abstract_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    abstract_table.autofit = False
    abstract_table.columns[0].width = Inches(6.5)
    
    cell = abstract_table.cell(0, 0)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)

    p_abs = cell.paragraphs[0]
    p_abs.paragraph_format.line_spacing = 1.15
    run_abs_title = p_abs.add_run("ÖZET — ")
    run_abs_title.bold = True
    run_abs_title.font.size = Pt(9.5)
    
    run_abs_text = p_abs.add_run(
        "Biyometrik yüz tanıma sistemleri; sınır güvenliği, uzaktan kimlik doğrulama (e-KYC) ve akıllı cihaz güvenliğinde kritik rol oynamaktadır. Ancak bu sistemler; basılı fotoğraf (print attack), ekran yansıtma (replay attack) ve yapay zeka tabanlı sentetik yüzler (deepfake) gibi fiziksel ve dijital sunum saldırılarına (Presentation Attacks) karşı savunmasızdır. Bu çalışmada, standart RGB kameralardan alınan yüz görüntülerinde mikro doku deformasyonlarını tespit eden derin öğrenme tabanlı bir anti-spoofing sistemi geliştirilmiştir. Google Xception derin evrişimli sinir ağı omurgası kullanılmış; iki aşamalı transfer öğrenme ve aşamalı ince ayar (Fine-Tuning) protokolü uygulanmıştır. Aşırı öğrenmeyi (overfitting) engellemek amacıyla veri artırma, küresel ortalama havuzlama ve seyreltme (%40 Dropout) entegre edilmiştir. Modelin kara kutu yapısını şeffaflaştırmak amacıyla Grad-CAM (Açıklanabilir Yapay Zeka - XAI) ısı haritası mekanizması sisteme dahil edilmiştir. Model, eğitimde hiç kullanılmayan 750 adet bağımsız test görselinde %93.33 genel doğruluk (Accuracy), %97.0 sahte tespit hassasiyeti (Precision) ve %97.4 gerçek insan duyarlılığı (Recall) elde etmiştir.\n\n"
    )
    run_abs_text.font.size = Pt(9.5)

    run_kw_title = p_abs.add_run("Anahtar Kelimeler — ")
    run_kw_title.bold = True
    run_kw_title.font.size = Pt(9)
    run_kw_text = p_abs.add_run("Biyometrik Güvenlik, Sahte Yüz Tespiti (Anti-Spoofing), Xception CNN, Transfer Öğrenme, Açıklanabilir Yapay Zeka (XAI), Grad-CAM, Deepfake.")
    run_kw_text.font.size = Pt(9)
    run_kw_text.font.italic = True

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # BÖLÜM EKLEME YARDIMCISI
    def add_section_heading(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(30, 58, 138) # Dark Blue
        return h

    def add_body_p(text, bold_prefix=None):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(10.5)
            r_pre.font.bold = True
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(30, 41, 59)
        return p

    # 1. GİRİŞ
    add_section_heading("1. GİRİŞ (INTRODUCTION)")
    add_body_p(
        "Biyometrik yüz doğrulama sistemleri, temas gerektirmeyen doğası ve yüksek kullanım ergonomisi nedeniyle dijital kimlik doğrulamanın temel standardı haline gelmiştir. Bununla birlikte klasik yüz tanıma sistemleri yalnızca bireyin kimlik eşleştirmesine odaklanmakta, kameraya gösterilen yüzün canlı bir insana mı yoksa bir ekrandan veya kağıttan yansıtılan bir sahteciliğe mi ait olduğunu ayırt edememektedir."
    )
    add_body_p(
        "Literatürde Sunum Saldırısı Tespiti (Presentation Attack Detection - PAD) olarak adlandırılan bu alanda çoğu ticari çözüm kızılötesi (IR) veya LiDAR sensörlerine dayanır. Ancak bu donanımlar standart bilgisayarlarda ve mobil aygıtlarda bulunmaz. Bu çalışmanın temel hedefi, standart 2D RGB kameralardan elde edilen tek bir kare görüntü üzerinden sahteciliği yakalayan, düşük parametreli ve yüksek doğruluklu bir derin öğrenme mimarisi sunmaktır."
    )

    # 2. MATERYAL VE YÖNTEM
    add_section_heading("2. MATERYAL VE YÖNTEM (MATERIALS AND METHODS)")
    add_body_p(
        "Çalışmada toplam 3.746 adet 256x256 çözünürlüğünde RGB yüz görselinden oluşan dengeli bir veri kümesi kullanılmıştır. Veri kümesi 2.996 eğitim (1.549 gerçek, 1.447 sahte) ve 750 bağımsız doğrulama görseli (388 gerçek, 362 sahte) olarak ayrılmıştır. Eğitim esnasında aşırı öğrenmeyi (overfitting) önlemek için ±30° döndürme, %20 yakınlaştırma ve yatay çevirme gibi veri artırma (Data Augmentation) işlemleri uygulanmıştır."
    )
    add_body_p(
        "Model omurgası olarak Google Xception (Extreme Inception) seçilmiştir. Xception mimarisi, standart evrişim işlemini derinlik ve uzamsal kanallara ayıran 'Derinlik Ayrılabilir Evrişimler' (Depthwise Separable Convolutions) kullanarak hesaplama karmaşıklığını ciddi biçimde düşürürken öznitelik yakalama kapasitesini artırır."
    )

    add_body_p("Eğitim Protokolü İki Aşamada Gerçekleştirilmiştir:", bold_prefix="• ")
    add_body_p("Aşama 1 (Isınma / Warm-up): Xception omurgası kilitlenmiş, rastgele ağırlıklarla başlayan yeni eklenen Dense(256) ve Sigmoid sınıflandırma başlığı 3 epoch boyunca lr=1e-4 ile eğitilmiştir.")
    add_body_p("Aşama 2 (İnce Ayar / Fine-Tuning): Xception'ın son 20 evrişim katmanının kilidi açılarak çok düşük bir öğrenme oranı (lr=1e-5) ile 10 epoch boyunca hedef veri kümesine hassas uyarlama yapılmıştır.")

    # 3. AÇIKLANABİLİR YAPAY ZEKA
    add_section_heading("3. AÇIKLANABİLİR YAPAY ZEKA (GRAD-CAM XAI)")
    add_body_p(
        "Derin öğrenme modelleri geleneksel olarak 'kara kutu' kabul edilir. Güvenlik kritik sistemlerde modelin bir kararı neden verdiğinin kanıtlanması gerekir. Bu projede Grad-CAM (Gradient-weighted Class Activation Mapping) algoritması sisteme entegre edilmiştir. Son evrişim katmanındaki gradyan akışları analiz edilerek yüz üzerindeki pikseller renklendirilmiş; modelin karar verirken ekran yansımaları, keskin kenarlar ve sentetik doku bozulmalarına odaklandığı görsel olarak kanıtlanmıştır."
    )

    # 4. DENEYSEL BULGULAR VE METRİKLER
    add_section_heading("4. DENEYSEL BULGULAR VE PERFORMANS ANALİZİ")
    add_body_p("Modelin daha önce hiç karşılaşmadığı 750 bağımsız test görseli üzerindeki performans karnesi aşağıda sunulmuştur:")

    # TABLO 1: Performans Tablosu
    table = doc.add_table(rows=4, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    headers = ["Sınıf", "Hassasiyet (Precision)", "Duyarlılık (Recall)", "F1-Skoru", "Test Görsel Sayısı"]
    for i, h in enumerate(headers):
        c = table.cell(0, i)
        set_cell_background(c, "1E3A8A") # Dark blue
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    data = [
        ["Gerçek (Class 0)", "%90.0", "%97.4", "0.94", "388"],
        ["Sahte (Class 1)", "%97.0", "%89.0", "0.93", "362"],
        ["Ağırlıklı Ortalama", "%94.0", "%93.3", "0.93", "750 (Genel Doğruluk: %93.33)"]
    ]

    for row_idx, row_data in enumerate(data, start=1):
        bg = "F1F5F9" if row_idx % 2 == 1 else "FFFFFF"
        if row_idx == 3:
            bg = "E2E8F0"
        for col_idx, text in enumerate(row_data):
            c = table.cell(row_idx, col_idx)
            set_cell_background(c, bg)
            set_cell_margins(c, top=70, bottom=70, left=100, right=100)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(text)
            r.font.size = Pt(9)
            if row_idx == 3:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # KARŞIKLIK MATRİSİ AÇIKLAMASI
    add_body_p(
        "Karışıklık Matrisi (Confusion Matrix) Sonuçları: 388 gerçek bireyin 378'i ilk bakışta doğru tespit edilmiş olup yanlış ret oranı (FRR) yalnızca %2.6'dır. 362 sahte girişimden 322'si başarıyla engellenmiştir. Modelin 'Sahte' dediği durumlarda doğruluk payı %97.0'dir."
    )

    # METRİK GRAFİĞİNİ EKLE (Varsa)
    if os.path.exists(config.METRICS_PLOT_PATH):
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.keep_with_next = True
        run_img = p_img.add_run()
        run_img.add_picture(config.METRICS_PLOT_PATH, width=Inches(5.5))
        
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(12)
        r_cap = cap.add_run("Şekil 1: Xception Modeli Eğitim ve Doğrulama Doğruluk/Kayıp Grafikleri (10 Epoch Fine-Tuning)")
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = RGBColor(100, 116, 139)

    # 5. SONUÇ VE GELECEK ÇALIŞMALAR
    add_section_heading("5. SONUÇ VE GELECEK ÇALIŞMALAR")
    add_body_p(
        "Bu çalışmada geliştirilen sistem; özel donanım sensörlerine ihtiyaç duymaksızın standart RGB kameralar üzerinden tek karede %93.33 başarıyla çalışan modern bir biyometrik güvenlik kalkanıdır. Grad-CAM ısı haritaları sayesinde kararların izlenebilirliği güvenceye alınmıştır. Gelecek çalışmalarda modelin TensorFlow Lite formatına kuantize edilerek mobil cihazlarda ve mikrodenetleyicilerde (Raspberry Pi) 30+ FPS canlı video akışında çalıştırılması hedeflenmektedir."
    )

    # 6. KAYNAKÇA
    add_section_heading("KAYNAKÇA (REFERENCES)")
    refs = [
        "[1] F. Chollet, 'Xception: Deep Learning with Depthwise Separable Convolutions,' IEEE CVPR, pp. 1251-1258, 2017.",
        "[2] R. R. Selvaraju et al., 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,' IEEE ICCV, pp. 618-626, 2017.",
        "[3] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, MIT Press, 2016.",
        "[4] Z. Boulkenafet, J. Komulainen, and A. Hadid, 'Face spoofing detection using colour texture analysis,' IEEE TIFS, vol. 11, no. 8, pp. 1818-1830, 2016.",
        "[5] Z. Yu et al., 'Searching central difference convolutional networks for face anti-spoofing,' IEEE CVPR, pp. 5295-5305, 2020."
    ]
    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_after = Pt(3)
        r_ref = p_ref.add_run(ref)
        r_ref.font.size = Pt(8.5)
        r_ref.font.color.rgb = RGBColor(71, 85, 105)

    doc.save(output_docx_path)
    print(f"[OK] Word belgesi başarıyla kaydedildi: {output_docx_path}")

def convert_docx_to_pdf(docx_path, pdf_path):
    print(">> Microsoft Word kullanılarak PDF üretiliyor...")
    abs_docx = os.path.abspath(docx_path)
    abs_pdf = os.path.abspath(pdf_path)

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0

    try:
        doc = word.Documents.Open(abs_docx)
        # 17 = wdExportFormatPDF
        doc.SaveAs(abs_pdf, FileFormat=17)
        doc.Close()
        print(f"[OK] PDF belgesi başarıyla kaydedildi: {abs_pdf}")
    except Exception as e:
        print(f"[HATA] PDF dönüştürme sırasında sorun oluştu: {e}")
    finally:
        word.Quit()

if __name__ == "__main__":
    out_docx = os.path.join(config.BASE_DIR, "KONFERANS_BILDIRISI_EMIRHAN_DAG.docx")
    out_pdf = os.path.join(config.BASE_DIR, "KONFERANS_BILDIRISI_EMIRHAN_DAG.pdf")
    
    create_academic_docx(out_docx)
    convert_docx_to_pdf(out_docx, out_pdf)
    print("\n>> İşlem tamamlandı! Word ve PDF belgeleriniz hazır.")
