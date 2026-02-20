import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

# Detta steget gjordes väldigt lätt av att läsa DESC i mnist. Dokumentera alla steg som de beskriver och försöka göra likadant här.
# Jag använde mig även av bilden på en femma som jag skrev ut i Mnist_Modeller för att se hur de faktiskt kan se ut.
# Det som var svårt var att få bort skuggor och liknande från bilder tagna på papper med telefonen. 
# Ville man bara köra predict mot handritade siffror på en dator så var det väldigt simpelt.
# Skuggorna kunde vara olika starka och behöva hanteras olika. Igenom att mixtra med parametrarna så hittade jag tillslut rätt
# Självklart kommer det alltid gå att lura modellen. Exempelvis om man skulle skriva en siffra med en röd bläckpenna
# Det skulle detta skriptet ha svårt att se. Programmet hade förmodligen filtrerat bort det som brus/skugga.
# Hade man velat så hade man kunnat utveckla appen ännu mer för att exempelvis vrida upp input siffror för att ta eventuella liggande siffror
# Man hade också kunnat göra appen mer användarvänlig och lagt in varningar för otydliga streck, beskrivningar av hur man ska skriva för bäst resultat osv.


def to_mnist_like_01(path: str):

    # Tar bilden och gör om den till en 2-d array
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32)  # 0..255

    # Sparar orginalet
    original = arr.copy()

    # Tar bort brus och skuggor kraftigt. Detta för att kunna läsa handskrivna  siffror papper med dålig tagna bilder.
    arr = np.clip(arr * 3, 0, 255)

     # Inverterar bilden om den är ljus istället för mörk
    if arr.mean() > 127:
        arr = 255.0 - arr
    
    # Skalar värdet på samma sätt som vi gjort i modellen
    arr01 = arr / 255.0

    # Binarisera 
    bw = (arr01 > 0.25).astype(np.float32)

    # Hittar vart siffran är på bilden
    coords = np.argwhere(bw > 0)

    # Klipper ut bilden
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = bw[y0:y1, x0:x1]

    # Resize till max 20 pixlar
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Här behövde vi skala tillbaka till gråskala för att kunna hantera det i Image
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))

    # BILINEAR hjälper att mjuka ut siffran när bilden skalas om till 20x20
    # Detta gör den mer lik siffrorna i mnist datan
    resized = cropped_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Gör om till numpy och skalar tillbaka till 0-1
    resized = np.array(resized).astype(np.float32) / 255.0

    # Pad till 28x28
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    # Center-of-mass
    # Centrerar bilden efter på samma sätt som mnist om man kollar i DESC
    cy, cx = ndimage.center_of_mass(canvas)
    shift_y = int(round(14 - cy)) if not np.isnan(cy) else 0
    shift_x = int(round(14 - cx)) if not np.isnan(cx) else 0
    canvas = ndimage.shift(canvas, shift=(shift_y, shift_x), order=1, mode="constant", cval=0.0)

    X = canvas.reshape(1, -1).astype(np.float32)

    return X, original, canvas