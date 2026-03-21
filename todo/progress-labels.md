# Pre-Step-Labels + Gesamtfortschritt

## 1. Pre-Step-Labels

Vor jedem Verarbeitungsschritt ein Label anzeigen, das beschreibt was passiert (z.B. "Loading model...", "Generating segment 1/3..."). Konsistent über alle Worker hinweg.

## 2. Gesamtfortschritt

Über alle Steps hinweg eine Gesamtprozent-Zahl, die den Fortschritt über alle Sub-Steps aggregiert. Nicht nur pro Worker, sondern über den gesamten generate.py-Aufruf.
