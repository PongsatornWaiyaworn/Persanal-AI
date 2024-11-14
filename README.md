# Personal-AI

| Class | Title                                              | Workshop                                                                 |
|-------|----------------------------------------------------|--------------------------------------------------------------------------|
|   1   | xPore: An AI for Bioinformaticians                 | [Gaussian Mixture Modelling](workshop/GMM.ipynb)                        |
|   2   | Learning from Biosignal                            | [1D CNN for Brain Signal](workshop/biosignals/)                          |
|   3   | AI for Detecting Code Plagiarism                   | [code2vec to Detect Code Clone](workshop/CodeCloneDetection.ipynbCodeCloneDetection.ipynb)      |
|   4   | BiTNet: AI for Diagnosing Ultrasound Image         | [NLP Classification](workshop/Image_classification_EfficientNetB5.ipynb) |
|   5   | Mental Disorder Detection from Social Media Data   | [Mental Disorder Detection](workshop/Mental_disorder_detection.ipynb)   |
|   6   | AI for Arresting Criminals                         | [YOLO V8 Experiment](workshop/)                                |

## Content

- [xPore: An AI for Bioinformaticians](#xpore-an-ai-for-bioinformaticians)
- [Learning from Biosignal](#learning-from-biosignal)
- [AI for Detecting Code Plagiarism](#ai-for-detecting-code-plagiarism)
[BiTNet: AI for Ultrasound Image Classification](#bitnet-ai-for-ultrasound-image-classification)
- [Mental Disorder Detection from Social Media Data](#mental-disorder-detection-from-social-media-data)
- [AI for Arresting Criminals](#ai-for-arresting-criminals)

---

### xPore: An AI for Bioinformaticians

1. **Problem Statement (การกำหนดปัญหา)**
   - งานวิจัยนี้มุ่งพัฒนาโมเดลตรวจจับการดัดแปลง RNA ประเภท m6A ซึ่งเป็นการเปลี่ยนแปลงทางพันธุกรรมที่เกี่ยวข้องกับกระบวนการทำงานในเซลล์และโรคต่าง ๆ โดยใช้เทคโนโลยี Nanopore sequencing ในการตรวจจับข้อมูล RNA sequencing โดยตรง

2. **Data Collection and Preparation (การเก็บรวบรวมและเตรียมข้อมูล)**
   - เก็บข้อมูลจาก Nanopore sequencing เช่น ไฟล์ FAST5 และ FASTQ สำหรับการตรวจจับการดัดแปลงใน RNA

3. **Machine Learning Modeling (การสร้างโมเดลด้วย Machine Learning)**
   - ใช้ Bayesian Gaussian Mixture Model (GMM) ในการวิเคราะห์ข้อมูล RNA แบบละเอียด เพื่อให้การทำนายแม่นยำและรวดเร็วขึ้น

4. **Evaluation (การประเมินผล)**
   - ประเมินผลด้วย ROC Curve, Precision-Recall Curve, และ Accuracy เพื่อวัดความแม่นยำในการตรวจจับ m6A

5. **Visualization and Presentation (การแสดงผลและนำเสนอ)**
   - นำเสนอผลด้วยการสร้างกราฟและพล็อต เช่น แผนภาพการทำงานของ xPore และการตรวจจับ m6A ใน RNA ทั่วทั้ง transcriptome

6. **Future Work (งานวิจัยในอนาคต)**
   - แนวทางในอนาคตคือการพัฒนาเทคนิคใหม่ เช่น Deep Autoencoder ร่วมกับ GMM เพื่อเพิ่มประสิทธิภาพในการตรวจจับการดัดแปลงทางพันธุกรรม

---

### Learning from Biosignal

1. **ความสำคัญของการวิเคราะห์ Biosignal**
   - การวิเคราะห์ Biosignal เช่น EEG, ECG, EMG ช่วยในการวินิจฉัยโรคและตรวจสอบสุขภาพ ซึ่งเป็นการวิเคราะห์เพื่อทำความเข้าใจการทำงานของร่างกายอย่างละเอียด

2. **ขั้นตอนสำคัญในการวิเคราะห์ Biosignal**
   - **การเตรียมข้อมูล (Preprocessing):** กำจัดเสียงรบกวนจากสัญญาณเพื่อความแม่นยำ
   - **การสกัดคุณสมบัติ (Feature Extraction):** สร้างตัวชี้วัดสำคัญจากสัญญาณ เช่น ความถี่และแอมพลิจูด
   - **การสร้างแบบจำลอง (Model Construction):** ใช้ Machine Learning เช่น CNN และ RNN ในการเรียนรู้ความสัมพันธ์ของข้อมูล

3. **การวิเคราะห์การนอนหลับ (Sleep Stage Scoring)**
   - ใช้ข้อมูล Polysomnogram (PSG) เพื่อวิเคราะห์การนอน เช่น Non-REM และ REM โดยประยุกต์ใช้ Deep Learning เพื่อประเมินการนอนอัตโนมัติ

4. **โมเดลที่ใช้ในการวิเคราะห์การนอน**
   - **DeepSleepNet:** โมเดลสองระดับ ใช้ CNN ในการสกัดข้อมูลและ RNN ในการเรียนรู้ลำดับการเปลี่ยนแปลงการนอน
   - **TinySleepNet:** โมเดลที่เล็กลง ลดการใช้ทรัพยากร แต่ยังมีประสิทธิภาพในการจำแนกช่วงการนอนได้ดี

5. **การประเมินผลโมเดล**
   - ใช้ k-fold cross-validation วัดค่า Accuracy, F1-Score, Cohen's Kappa และ Hypnogram เพื่อประเมินผลในทุกช่วงการนอน

6. **ความท้าทายและแนวทางการพัฒนาในอนาคต**
   - พัฒนาระบบที่ใช้กับอุปกรณ์สวมใส่ เช่น Smart Eye-Mask สำหรับตรวจสอบสุขภาพจากระยะไกล และเทคนิค Transfer Learning เพื่อปรับใช้โมเดลในสภาพแวดล้อมต่าง ๆ

---

### AI for Detecting Code Plagiarism

1. **ความสำคัญและประเภทของ Code Clones**
   - **Code Clones** คือกลุ่มโค้ดที่มีลักษณะหรือโครงสร้างที่คล้ายกัน แบ่งเป็น 4 ประเภท:
     - **Type 1:** โค้ดที่เหมือนกันทั้งหมด ยกเว้นการจัดวางและคอมเมนต์
     - **Type 2:** โค้ดที่เหมือนกันยกเว้นค่าคงที่ ตัวแปร และชนิดข้อมูล
     - **Type 3:** โค้ดที่คล้ายกัน แต่มีการแก้ไขบางบรรทัด
     - **Type 4:** โค้ดที่มีผลลัพธ์เหมือนกัน แต่ต่างกันที่การเขียนโค้ดหรืออัลกอริทึม

2. **ความท้าทายและความจำเป็นในการตรวจจับ Code Clones**
   - ปัญหาจาก Code Clones เพิ่มความซับซ้อนของโค้ดและลดคุณภาพ การจัดการ Code Clones ช่วยลดความซ้ำซ้อนและปรับปรุงคุณภาพของโค้ด

3. **กระบวนการและเทคนิคในการตรวจจับ Code Clones**
   - **Preprocessing:** เตรียมข้อมูลโค้ด เช่น การจัดรูปแบบและการกรอง
   - **Transformation และ Matching:** การเปลี่ยนรูปแบบโค้ดเพื่อค้นหาความคล้ายคลึง
   - **Post Processing:** รวบรวมผลลัพธ์การตรวจจับ

4. **การสร้าง Machine Learning Model เพื่อใช้ในระบบตรวจจับโค้ดคล้าย (Merry Engine)**
   - **การเตรียมข้อมูล:** ใช้ BigCloneBench แหล่งข้อมูลโค้ดคล้าย
   - **ดึงค่า Metrics:** คำนวณค่าทั้งเชิงโครงสร้าง (syntactic) และความหมาย (semantic) ด้วย code2vec
   - **การฝึกโมเดล:** ใช้โมเดลเช่น Decision Tree, Random Forest, และ SVM

5. **การประเมินผลและประสิทธิภาพของ Merry Engine**
   - ประเมินโมเดลบน BigCloneBench และซอฟต์แวร์จริง เช่น Precision, Recall และ F1-score โดยทดสอบในกลุ่มผู้ใช้งาน

6. **บทสรุปและแนวทางการพัฒนาต่อ**
   - Merry Web-Based Code Clone Detection Tool พัฒนามาเพื่อความแม่นยำและใช้งานง่ายในอนาคตควรรองรับภาษาโปรแกรมอื่น ๆ และปรับปรุงประสิทธิภาพของ code2vec

---

### BiTNet: AI for Ultrasound Image Classification  

1. **การเตรียม Dataset (เน้นที่ภาพอัลตราซาวด์)**
   - **การตั้งชื่อไฟล์:** จัดการข้อมูลให้สอดคล้องกับลักษณะดังนี้:  
     - Classes, Viewing Angle และ Patient Case เป็นต้น

   - **การลบข้อมูลพื้นหลัง (Remove BG Information):**  
     - ลบหรือครอบตัดส่วนที่ไม่เกี่ยวข้อง เช่น พื้นหลังของภาพ เพื่อเน้นบริเวณตับ  

   - **การปรับขนาดภาพ (Input Size):**  
     - มาตรฐานภาพอินพุต เช่น **224x224** หรือ **299x299 pixels**  

   - **การเพิ่มข้อมูล (Data Augmentation):**  
     ใช้เทคนิคสร้างข้อมูลเพิ่มเติมเพื่อเพิ่มความหลากหลาย:  
     - **Horizontal/Vertical Shift:** เลื่อนภาพในแนวนอนและแนวตั้ง  
     - **Rotation 30°:** หมุนภาพ  
     - **Brightness Adjustment:** ปรับความสว่าง  
     - **Shear/Zoom:** บิดและซูมภาพ  
     - **No Flip:** หลีกเลี่ยงการพลิกภาพเพื่อคงลักษณะตำแหน่งอวัยวะ  

2. **การพัฒนาโมเดล (Model Development)**  
   - **EfficientNet (Base Model) vs BiTNet:**  
     - **EfficientNet:** โมเดลที่ปรับสมดุลระหว่างขนาดและประสิทธิภาพ เหมาะกับข้อมูลขนาดเล็ก  
     - **BiTNet:** ปรับปรุงจาก EfficientNet โดยเน้นการตรวจจับลักษณะเฉพาะในภาพอัลตราซาวด์ เช่น เนื้อเยื่อตับที่มีพยาธิ  

   - **การประยุกต์ใช้งาน (Applications):**  
     1. **Auto Pre-screening:**  
        - ใช้ AI คัดกรองเบื้องต้น ลดภาระการทำงานของแพทย์  
     2. **Assisting Tool:**  
        - เป็นเครื่องมือช่วยแพทย์ในการวินิจฉัยโรค  

3. **การฝึกโมเดล (Training Pre-trained Models)**  
   - **Freezed Layers:**  
     - ล็อกเลเยอร์บางส่วนของโมเดล Pre-trained เพื่อประหยัดเวลาและทรัพยากร  

   - **Unfreezed Layers:**  
     - ปลดล็อกเลเยอร์ทั้งหมด เพื่อปรับการเรียนรู้ให้เข้ากับชุดข้อมูลใหม่  

4. **การประเมินผล (Evaluation)**  
   - **การเปรียบเทียบโมเดล:**  
     - ทดสอบ BiTNet และ EfficientNet ด้วยชุดข้อมูลเดียวกัน  

   - **การประเมินผลสำหรับ 2 Applications:**  
     1. **Auto Pre-screening:**  
        - ตรวจสอบความแม่นยำในการคัดกรองข้อมูล  
     2. **Assisting Tool:**  
        - วัดผลลัพธ์ของผู้ใช้งานเมื่อใช้/ไม่ใช้ AI  

   - **การทดสอบทางสถิติ:**  
     1. **Independent Samples T-Test:**  
        - เปรียบเทียบค่าความแตกต่างเฉลี่ยของความมั่นใจระหว่างการทำนายที่ถูก/ผิดใน BiTNet และ EfficientNet  
        - **Hypothesis:** BiTNet มีความมั่นใจที่แตกต่างเด่นชัดมากกว่า EfficientNet  

     2. **Paired Samples T-Test:**  
        - เปรียบเทียบประสิทธิภาพ (Accuracy, Precision, Recall) ของแพทย์  
          - **Hypothesis 1:** ผู้ใช้ที่ได้รับ AI ช่วยมีผลลัพธ์ที่ดีกว่า  
          - **Hypothesis 2:** ความแม่นยำของแพทย์ในรอบแรกและรอบสองไม่มีความแตกต่าง  
        - วิเคราะห์ความสอดคล้องระหว่างคำแนะนำ AI กับการตัดสินใจของแพทย์  
          - **Hypothesis:** ความสอดคล้องเพิ่มขึ้นเมื่อมี AI ช่วย  

5. **การแสดงผล (Visualization)**  
   - **การเปรียบเทียบโมเดล:**  
     - กราฟแสดงค่า Accuracy, Precision, Recall ระหว่าง BiTNet และ EfficientNet  

   - **การใช้งานจริง:**  
     1. **Auto Pre-screening:**  
        - Visualization ของ Confidence Score จาก AI  
     2. **Assisting Tool:**  
        - ตารางเปรียบเทียบผลลัพธ์ของแพทย์เมื่อใช้และไม่ใช้ AI  

   - **การทดสอบทางสถิติ:**  
     - กราฟและตารางแสดงค่าทางสถิติ เช่น T-value และ P-value  


---

### Mental Disorder Detection from Social Media Data

1. **Introduction**
   - **โซเชียลมีเดียในฐานะแหล่งข้อมูล (Social Media as Data Source)**

     **คุณลักษณะของโซเชียลมีเดีย:**
     โซเชียลมีเดียเป็นข้อมูลที่ไม่เป็นโครงสร้าง (Unstructured Data) ที่มีลักษณะต่างจากข้อมูลทั่วไปในระบบสุขภาพ โดยมีองค์ประกอบ เช่น
     - **Dynamic Nature:** เนื้อหาเปลี่ยนแปลงรวดเร็วตามพฤติกรรมผู้ใช้
     - **Multimodal Data:** มีทั้งข้อความ รูปภาพ วิดีโอ และอื่นๆ

     **ผลกระทบต่อสุขภาพจิต:**
     งานวิจัยจาก Journal of Medical Internet Research ระบุว่า โพสต์ในโซเชียลมีเดียสะท้อนอารมณ์ ความรู้สึก และความเครียดที่สัมพันธ์กับภาวะซึมเศร้าและโรควิตกกังวล

2. **Data Collection**
   - **วิธีการเก็บรวบรวมข้อมูล (Methods of Data Collection)**

     **Direct Data Collection:**
     - **การเก็บข้อมูลผ่าน แบบสอบถาม (Questionnaires):**
       ใช้เครื่องมือเช่น Patient Health Questionnaire-9 (PHQ-9) เพื่อวัดระดับภาวะซึมเศร้า
     - **การใช้ Electronic Health Records (EHR):**
       รวบรวมข้อมูลสุขภาพจิตจากคลินิกหรือโรงพยาบาล

     **Passive Data Collection:**
     - ใช้เทคนิคการรวบรวมข้อมูลจากโพสต์บนแพลตฟอร์ม เช่น Twitter หรือ Reddit
     - **ตัวอย่างคำค้นหา:** "I feel depressed," "I'm diagnosed with anxiety"

     **ประเด็นด้านจริยธรรม (Ethical Considerations):**
     - **ความเป็นส่วนตัว:** การรวบรวมข้อมูลที่มีการลบตัวระบุข้อมูลส่วนบุคคล (De-identification)
     - **การขออนุญาต:** ต้องได้รับการยินยอมจากผู้ใช้หากเป็นการเก็บข้อมูลโดยตรง

3. **Data Preprocessing**
   - **การเตรียมข้อมูล (Preprocessing Techniques)**

     - **Tokenization:** แยกคำหรือประโยคออกเป็นหน่วยย่อย
     - **Stop Word Removal:** ลบคำที่ไม่สำคัญ เช่น “is,” “the,” “and”
     - **Vectorization:**
       ใช้ TF-IDF (Term Frequency-Inverse Document Frequency): วัดความสำคัญของคำในเอกสาร

       **ตัวอย่างการใช้ Python:**
       ```python
       from sklearn.feature_extraction.text import TfidfVectorizer
       vectorizer = TfidfVectorizer()
       X = vectorizer.fit_transform(corpus)
       ```

     - **การทำ Annotation:**
       Annotators ที่มีความเชี่ยวชาญสามารถระบุโพสต์ที่สอดคล้องกับคำถามเชิงคลินิก เช่น การพูดถึงความรู้สึกสิ้นหวัง

4. **Predictive Modeling**
   - **เทคนิคการสร้างโมเดล (Modeling Techniques)**

     **Supervised Learning Algorithms:**
     - ใช้โมเดลเช่น Logistic Regression, SVM (Support Vector Machines), และ Neural Networks

     **ตัวอย่างการสร้างโมเดล:**
     ```python
     from sklearn.linear_model import LogisticRegression
     model = LogisticRegression()
     model.fit(X_train, y_train)
     predictions = model.predict(X_test)
     ```

     **NLP-based Models:**
     - **BERT (Bidirectional Encoder Representations from Transformers):**
       โมเดลการเรียนรู้เชิงลึกสำหรับการประมวลผลภาษา

     **ฟีเจอร์ที่สำคัญ (Key Features):**
     - การวิเคราะห์โครงสร้างประโยค (Syntax)
     - การวิเคราะห์อารมณ์ (Sentiment Analysis)
     - การใช้เครื่องมือเช่น LIWC (Linguistic Inquiry and Word Count)

5. **Evaluation**
   - **ตัวชี้วัดประสิทธิภาพ (Evaluation Metrics)**

     - **Precision:** ความถูกต้องของผลลัพธ์ที่โมเดลทำนายว่าเป็นบวก
     - **Recall:** ความสามารถของโมเดลในการตรวจจับข้อมูลที่เป็นบวกทั้งหมด
     - **F1-Score:** ค่าเฉลี่ยเชิงเรขาคณิตระหว่าง Precision และ Recall

     **Baseline Comparison:**
     - เปรียบเทียบผลลัพธ์ของโมเดลกับข้อมูลจริง (Ground Truth)

6. **Challenges and Future Directions**
   - **ข้อท้าทาย (Challenges):**
     - ความซับซ้อนของภาษาในโซเชียลมีเดีย เช่น การใช้แสลง (Slangs)
     - ความหลากหลายของข้อมูล เช่น ภาษาที่แตกต่าง หรือบริบทที่ไม่ชัดเจน

   - **เป้าหมายในอนาคต (Future Goals):**
     - พัฒนาระบบอัตโนมัติสำหรับการคัดกรองผู้มีความเสี่ยง
     - ขยายการวิเคราะห์ไปยังข้อมูลหลายช่องทาง เช่น วิดีโอและเสียง
     - ใช้โมเดลที่รองรับ Multimodal Data เช่น Visual BERT


---

### AI for Arresting Criminals

1. **ภาพรวมของ Object Detection**

   **คำจำกัดความ:**
   การตรวจจับวัตถุ (Object Detection) คือการระบุที่ตั้ง (Localization) และจำแนกประเภท (Classification) ของวัตถุภายในภาพ  
   วัตถุประสงค์หลัก:
   - **Localization**: หาตำแหน่งวัตถุในภาพโดยใช้กรอบสี่เหลี่ยม (Bounding Box)
   - **Classification**: ระบุว่าวัตถุนั้นคืออะไร

   **ความสำคัญ:**
   - การตรวจจับวัตถุแบบเรียลไทม์เลียนแบบความสามารถของมนุษย์ในการวิเคราะห์ภาพ
   - มีบทบาทสำคัญในระบบยานยนต์ไร้คนขับ, ระบบเฝ้าระวัง, และหุ่นยนต์ ที่ต้องการความเร็วและความแม่นยำสูง

2. **YOLO คืออะไร**

   - YOLO ย่อมาจาก “You Only Look Once”  
     เน้นการตรวจจับวัตถุในภาพด้วยการประมวลผลเพียงครั้งเดียวผ่านโครงข่ายประสาทเทียม
   - **คุณสมบัติเด่น**:
     - ความเร็วสูง: เหมาะสำหรับการใช้งานแบบเรียลไทม์
     - การทำงานในรูปแบบเดียว: รวมการตรวจจับและจำแนกในขั้นตอนเดียว

3. **กระบวนการทำงานของ YOLO**

   - **การแบ่งภาพ:**  
      - ภาพอินพุตถูกแบ่งออกเป็นกริดขนาด \( S 	imes S \)
      - แต่ละช่องกริดจะรับผิดชอบการทำนาย \( B \) กรอบสี่เหลี่ยมและความน่าจะเป็นของคลาส \( C \)

   - **ผลลัพธ์การทำนาย:**
      - กรอบสี่เหลี่ยมแต่ละกรอบประกอบด้วย:
        - ค่าพิกัด (\( x, y \)), ความกว้าง (\( w \)), ความสูง (\( h \))
        - ค่า Confidence Score (\( P(Object) \)) คือความน่าจะเป็นที่กรอบนั้นมีวัตถุอยู่

   - **สถาปัตยกรรม:**
      - ใช้ Convolutional Neural Network (CNN) ในการประมวลผล
      - ตัวอย่าง: หาก \( S = 7 \), \( B = 2 \), \( C = 20 \)  
        ผลลัพธ์จะเป็น \( 7 	imes 7 	imes 30 \)

4. **วิวัฒนาการของ YOLO**

   **YOLOv1**:
   - รุ่นแรกที่นำเสนอโครงสร้างการตรวจจับแบบรวม
   - รวดเร็วแต่ยังมีข้อจำกัดในกรณีของวัตถุขนาดเล็กหรือที่ทับซ้อนกัน

   **YOLOv2 (YOLO9000)**:
   - เพิ่มการใช้ Anchor Box และ Batch Normalization
   - รองรับคลาสวัตถุสูงถึง 9,000 คลาส

   **YOLOv3**:
   - เพิ่มการทำนายหลายสเกล (Multi-scale Predictions)
   - ใช้ Darknet-53 ในการดึงคุณสมบัติ

   **YOLOv4**:
   - เพิ่มประสิทธิภาพด้วย CSPNet (Cross-Stage Partial Networks)
   - เหมาะสำหรับวัตถุขนาดเล็ก

   **YOLOv5**:
   - โดดเด่นในด้านความเร็วและใช้งานง่าย
   - มีน้ำหนักแบบ Pre-trained สำหรับการใช้งานจริง

   **YOLOv8**:
   - ใช้ **C2f Module** (Cross-Stage Partial Bottleneck with Two Convolutions):
     - รวมคุณลักษณะระดับสูงกับข้อมูลบริบทเพื่อเพิ่มความแม่นยำ
     - รองรับการใช้งานแบบถอยหลังกับ YOLOv5 ได้

5. **ขั้นตอนการทำงานของ YOLO**

   - **การฝึกโมเดล (Training):**
      - อินพุต: ภาพพร้อมข้อมูลที่ระบุกรอบและคลาส
      - ฟังก์ชันความสูญเสีย: ปรับให้เหมาะสมทั้งการระบุตำแหน่งและการจำแนกประเภท

   - **การทำนาย (Prediction):**
      - ผ่านโครงข่ายประสาทเทียมเพียงครั้งเดียว
      - ใช้ Non-Maximal Suppression (NMS) กรองกรอบที่ไม่จำเป็น:
        - กรองกรอบที่มีค่า Confidence ต่ำ
        - เลือกกรอบที่มีความมั่นใจสูงสุดในกรณีของการทับซ้อน

   - **การประเมินผล (Evaluation):**
      - ใช้เมตริกเช่น mean Average Precision (mAP)

6. **จุดเด่นและข้อจำกัดของ YOLO**

   **จุดเด่น:**
   - **ความเร็ว**: เหมาะสำหรับงานแบบเรียลไทม์
   - **ความเรียบง่าย**: โครงสร้างเข้าใจง่ายและใช้งานสะดวก
   - **ความยืดหยุ่น**: ใช้ได้หลากหลายโดเมน เช่น การเฝ้าระวังและยานยนต์ไร้คนขับ

   **ข้อจำกัด:**
   - ประสิทธิภาพลดลงในกรณีของวัตถุที่ทับซ้อนกัน
   - การตรวจจับขึ้นอยู่กับขนาดกริด ซึ่งอาจพลาดรายละเอียดเล็กๆ

7. **การประยุกต์ใช้งาน YOLO**

   - **ระบบเฝ้าระวัง (Surveillance):**
      - ตรวจจับกิจกรรมต้องสงสัยและแจ้งเตือนตำรวจ
      - ตัวอย่าง: ระบบ AI ในกล้องวงจรปิดที่ประเทศไทยเพื่อเฝ้าระวังบุคคลและยานพาหนะ

   - **ยานยนต์ไร้คนขับ (Autonomous Vehicles):**
      - ตรวจจับคนเดินถนน, รถยนต์ และสัญญาณจราจร

   - **การแพทย์ (Healthcare):**
      - ระบุเครื่องมือแพทย์หรือวิเคราะห์ภาพถ่ายทางการแพทย์

8. **กิจกรรมปฏิบัติ: YOLOv8**

   **วัตถุประสงค์:**  
   จำแนกวัตถุ 4 ประเภท ได้แก่ รถบัส, แท็กซี่, รถยนต์, และคนเดินถนน

   **ขั้นตอน:**
   1. แยกเฟรมจากวิดีโอ
   2. ติดป้ายกำกับแต่ละเฟรมด้วยกรอบสี่เหลี่ยม
   3. ฝึกโมเดล YOLOv8 ด้วยข้อมูลที่ติดป้ายกำกับ

   **เครื่องมือ:**
   - Python, Google Colab และ YOLOv8 Pre-trained Model
