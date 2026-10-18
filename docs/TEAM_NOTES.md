# Vec3 — Benchmark Chroma vs PostgreSQL/pgvector

**Μάθημα:** Συστήματα Διαχείρισης Πληροφορίας  
**Εξάμηνο:** Χειμερινό 2026–2027  
**Θέμα:** Comparison between Vector Databases — **Vec3: Chroma, pgvector**  
**Κατάσταση καταγραφής:** 18 Οκτωβρίου 2026

## 1. Στόχος

Συγκριτική αξιολόγηση δύο open-source vector database συστημάτων, **Chroma** και **PostgreSQL με pgvector**, στο ίδιο τοπικό μηχάνημα. Η εργασία καλύπτει εγκατάσταση, εισαγωγή vectors, indexes, similarity queries (με και χωρίς metadata filtering), απόδοση, κατανάλωση πόρων και πειράματα με περιορισμένη διαθέσιμη RAM.

## 2. Προδιαγραφές υπολογιστή (μετρημένες)

Τα παρακάτω προέρχονται από τις εξόδους εντολών `lscpu`, `free -h`, `lsblk`, `df -h /`, `cat /etc/os-release`, `uname -r` και ελέγχους εργαλείων που εκτελέστηκαν στο μηχάνημα.

| Κατηγορία | Τιμή |
| --- | --- |
| Μηχάνημα | Laptop (`hplaptop`, hostname κατά τον έλεγχο) |
| Λειτουργικό σύστημα | Debian GNU/Linux 13.2 (Trixie), x86_64 |
| Kernel | `6.12.57+deb13-amd64` |
| Επεξεργαστής | Intel Core i7-8550U @ 1.80 GHz |
| CPU cores / threads | 4 φυσικοί πυρήνες / 8 λογικοί επεξεργαστές |
| Αναφερόμενη μέγιστη CPU συχνότητα | 4.00 GHz |
| SIMD | AVX2 (επίσης AVX, FMA, SSE4.1/4.2) |
| CPU L3 cache | 8 MiB |
| RAM | 15 GiB συνολικά, περίπου 12 GiB `available` κατά τον έλεγχο |
| Swap | 15 GiB, 0 B χρησιμοποιημένα κατά τον έλεγχο |
| Φυσικός δίσκος | `/dev/sda`, 931.5 GiB, `ROTA=0` (μη περιστρεφόμενος/SSD) |
| Root partition | `/dev/sda6`, mounted στο `/` |
| Root filesystem | 419 GiB συνολικά, 51 GiB χρησιμοποιημένα, **348 GiB ελεύθερα** (13% χρήση) |
| Docker Engine | `26.1.5+dfsg1` (daemon προσβάσιμος) |
| Docker Compose | `docker-compose` **2.26.1-4** (standalone εντολή με παύλα) |
| Docker Compose plugin | Η εντολή `docker compose` **δεν** ήταν διαθέσιμη |
| Docker storage driver | `overlay2` |
| Docker cgroups | Driver: `systemd`, Version: **2** |
| Python | `3.13.5` |
| pip | `25.1.1` |
| Git | `2.47.3` |

**Σημειώσεις:**
- Το `available` RAM και ο ελεύθερος χώρος δίσκου αποτελούν στιγμιότυπο, όχι σταθερές τιμές.
- Ο CPU είναι laptop χαμηλής κατανάλωσης: κατά τα μεγάλα benchmarks καταγράφουμε συνθήκες, φόρτο και ενδεχόμενο thermal throttling.
- Το Docker υποστηρίζει cgroups v2, χρήσιμο για πειράματα με memory limits. Η συμπεριφορά του περιορισμού θα επαληθευτεί πρακτικά.
- Δεν απαιτείται GPU από την εκφώνηση του συγκεκριμένου project.

## 3. Τι έχει γίνει ως τώρα (επιβεβαιωμένο)

- [x] Έλεγχος CPU, RAM, swap και δίσκου.
- [x] Έλεγχος Debian, kernel, Python, pip, Git, Docker και cgroups.
- [x] Επιβεβαίωση ότι το `docker-compose --version` επιστρέφει `Docker Compose version 2.26.1-4`.
- [x] Δημιουργία τοπικού Git repository με `git init` στον φάκελο `~/vec3-benchmark`.
- [ ] **Δεν έχει επιβεβαιωθεί** σύνδεση με απομακρυσμένο GitHub repository (`origin`).
- [ ] **Δεν έχει επιβεβαιωθεί** αλλαγή του κλάδου σε `main` ή πρώτο commit.
- [ ] **Δεν έχει επιβεβαιωθεί** δημιουργία `compose.yaml`, εκκίνηση Chroma ή PostgreSQL/pgvector.
- [ ] **Δεν έχουν εκτελεστεί** benchmarks.

> Προσοχή: Έχουν προταθεί σε προηγούμενα βήματα φάκελοι και ένα Compose configuration, αλλά δεν έχουμε δει ακόμη την εκτέλεση των αντίστοιχων εντολών. Δεν τα καταγράφουμε ως ολοκληρωμένα.

## 4. Δομή του project

### 4.1 Επιβεβαιωμένη τοπική δομή πριν προστεθεί αυτό το README

```text
~/vec3-benchmark/
└── .git/                  # Δημιουργήθηκε με git init
```

### 4.2 Προτεινόμενη δομή όταν προχωρήσει η εργασία

```text
vec3-benchmark/
├── .gitignore             # Αποφυγή commit μεγάλων δεδομένων / μυστικών
├── README.md              # Τεκμηρίωση project & οδηγιών
├── compose.yaml           # Υπηρεσίες Chroma / PostgreSQL+pgvector
├── config/                # Παραμετροποίηση και pinned versions
├── datasets/              # Περιγραφές datasets, όχι απαραίτητα τα μεγάλα αρχεία στο Git
├── scripts/               # Ingestion, ground truth, benchmarks, analysis
└── results/               # Συγκεντρωτικά αποτελέσματα και διαγράμματα
```

Η προτεινόμενη δομή **δεν** είναι ακόμα επιβεβαιωμένα εγκατεστημένη στο laptop.

## 5. Κύριες απαιτήσεις της εκφώνησης

1. Εγκατάσταση **Chroma** και **pgvector** σε single-node περιβάλλον.
2. Ίδια datasets και queries στις δύο βάσεις, διαφορετικές διαστατικότητες, τουλάχιστον ένα dataset σύγχρονων text embeddings **384–1024 διαστάσεων**. Κλίμακα έως περίπου εκατομμύρια vectors ανά dataset, εφόσον το επιτρέπουν οι πόροι.
3. Μετρήσεις χρόνου εισαγωγής για μικρό/μεσαίο/τελικό μέγεθος, χρόνου δημιουργίας index, αποθηκευτικού χώρου και peak RAM.
4. Similarity search με κοινά υποστηριζόμενα distance metrics και φίλτρα επιλεκτικότητας **50%, 10%, 1%**· χρήση/υπολογισμός exact ground truth.
5. Σύγκριση σε **recall@10**, latency **p50/p95/p99**, QPS, CPU/RAM, με **1 και 8 ταυτόχρονους clients**.
6. Καμπύλες **recall–QPS** με περίπου **4–6** τιμές παραμέτρων αναζήτησης· σύγκριση με αντίστοιχο recall **0.90** και **0.95** αντί μόνο default configurations.
7. Τουλάχιστον ένα πείραμα με περιορισμό διαθέσιμης μνήμης κάτω από το μέγεθος δεδομένων + index.
8. Τουλάχιστον **5 επαναλήψεις** για κάθε timing, αναφορά median/mean με 95% CI ή τυπική απόκλιση, ορισμός cold/warm runs και warm-up, fixed random seeds.
9. Ακριβείς εκδόσεις λογισμικού, reproducible scripts και **repository URL + Git tag ή commit hash** στην τελική αναφορά.

**Πηγή:** `Class_Projects_2026-27.pdf`, σελίδες 2–4, ενότητες «General requirements» και «Comparison between Vector DataBases».

## 6. Επόμενα βήματα

1. Αντιγραφή αυτού του αρχείου στο `~/vec3-benchmark/README.md` και δημιουργία `.gitignore`.
2. Προαιρετική δημιουργία των φακέλων `config/`, `datasets/`, `scripts/`, `results/`.
3. Μετονομασία του κλάδου σε `main`, πρώτο τοπικό commit.
4. Δημιουργία **private** repository `vec3-benchmark` στο GitHub **χωρίς** αυτόματη δημιουργία README/.gitignore/license, ώστε να μην υπάρξει σύγκρουση ιστορικού.
5. Σύνδεση `origin` και `git push -u origin main`.
6. Σχεδίαση και έλεγχος Compose setup, καταγραφή **ακριβών** versions/images, εκκίνηση και health checks Chroma και PostgreSQL/pgvector.
7. Python environment με pinned dependencies, μικρό smoke test και σταδιακή ανάπτυξη benchmark harness.

## 7. Χρήσιμες εντολές αποτύπωσης κατάστασης

```bash
cd ~/vec3-benchmark
pwd
git status
git branch --show-current
git remote -v
find . -maxdepth 2 -not -path './.git/*' -not -path './.git' -print | sort

docker-compose --version
```

---

*Το παρόν README είναι αρχική αποτύπωση. Κάθε νέα εγκατάσταση, dataset, παράμετρος και αποτέλεσμα πρέπει να προστίθεται αφού επιβεβαιωθεί στο μηχάνημα.*
