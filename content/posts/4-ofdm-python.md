---
title: "Pythonic Example for OFDM"
date: "2019-10-5"
summary: "Orthogonal Frequency Modulation Division (OFDM), is a popular multi carrier transmission system applied to many common wireless systems. Here, we show a simple python example of OFDM."
math: true 
---

We will uncover the basic building blocks of an OFDM system at the transmitter and receiver side. OFDM (Orthogonal frequency division multiplexing) is a multicarrier system that is applied in a wide range of wireless transmission systems, such as LTE, WiMAX and DVB-T and DAB. The fundamental concept of a multicarrier system is the division of a high-rate transmitted data stream into several low-rate narrow subcarriers. This way, several advantages are obtained:

- Since the symbol duration is inverse proportional to the symbol rate, each subcarrier has relatively long symbols. Long symbols are robust against multipath fading, as it occurs in wireless systems.
- When a carrier is in a deep fade due to frequency-selectivity of the channel (i.e. the received energy on this carrier is very low), only the data on this subcarrier is lost, instead of the whole stream.
- Multicarrier systems allow easy multi-user resource sharing by allocating different subcarriers to different users.

Consider the following block diagram, which contains fundamental blocks for the OFDM system:

<!-- 
```python
%%tikz -l positioning,arrows --size 800,300
\tikzset{block/.style={draw,thick,minimum width=1cm,minimum height=1cm,align=center}}
\tikzset{node distance=0.5cm}
\tikzset{double distance=1pt}
\tikzset{>=latex}

\begin{scope}
\draw [->] (-0.5,0) node [left] {$\vec{b}$}  -- (0,0) node (SP1) [right,block] {S/P}; 
\node (M) [block,right=of SP1] {Mapping};
\node (IDFT) [block,right=of M] {IDFT};
\node (PS1) [block,right=of IDFT] {P/S};
\node (CP) [block,right=of PS1] {Add CP};

\draw [double,->] (PS1) -- (CP);

\def\lines{
\draw \ls ([yshift=3.mm]\from.east) -- ([yshift=3.mm]\to.west); 
\draw \ls ([yshift=1.mm]\from.east) -- ([yshift=1.mm]\to.west); 
\draw \ls ([yshift=-1.mm]\from.east) -- ([yshift=-1.mm]\to.west); 
\draw \ls ([yshift=-3.mm]\from.east) -- ([yshift=-3.mm]\to.west);   
}

\def\ls{[->]}
\def\from{SP1} \def\to{M} \lines;
\def\ls{[->,double]}
\def\from{M} \def\to{IDFT} \lines;
\def\from{IDFT} \def\to{PS1} \lines;



\node (C) [block,below right=of CP] {Channel}; 

\node (CP1) [block,below left=of C] {Remove CP};
\node (SP2) [block,left=of CP1] {S/P};
\node (DFT) [block,left=of SP2] {DFT};
\node (EQ) [block,left=of DFT] {Equalize};
\node (CE) [block,below=of EQ] {Channel\\Estimate};
\node (Dem) [block,left=of EQ] {Demapping};
\node (PS2) [block,left=of Dem] {P/S};
\draw [->] (PS2.west) -- +(-0.5,0) node [left] {$\hat{b}$};

\def\ls{[<-,double]}
\def\from{SP2} \def\to{CP1} \lines;
\def\from{DFT} \def\to{SP2} \lines;
\def\from{EQ} \def\to{DFT} \lines;
\def\from{Dem} \def\to{EQ} \lines;
\def\ls{[<-]}
\def\from{PS2} \def\to{Dem} \lines;

\draw [->,double,thick] (DFT.south) |- (CE.east);
\draw [->,double,thick] (CE.north) -- (EQ.south);

\draw [->,double] (CP) -| (C);
\draw [->,double] (C) |- (CP1);

\end{scope}
``` -->


![png](/posts/4-ofdm-python_files/4-ofdm-python_2_0.png)


In the following OFDM example, we will go through each block and describe its operation. However, before let us define some parameters that are used for the OFDM system:

The number of subcarriers $K$ describes, how many subcarriers are available in the OFDM system.


```python
K = 64 # number of OFDM subcarriers
```

The length of the cyclic prefix (CP) denotes the number of samples that are copied from the end of the modulated block to the beginning, to yield a cyclic extension of the block. There is a dedicated article on the CP of OFDM which treats its application in more detail.


```python
CP = K//4  # length of the cyclic prefix: 25% of the block
```

The number of pilots $P$ in the OFDM symbol describes, how many carriers are used to transmit known information (i.e. pilots). Pilots will be used at the receiver to estimate the wireless channel between transmitter and receiver. Further, we also define the value that each pilots transmits (which is known to the receiver).


```python
P = 8 # number of pilot carriers per OFDM block
pilotValue = 3+3j # The known value each pilot transmits
```

Now, let us define some index sets that describe which carriers transmit pilots and which carriers contain payload.


```python
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.

# For convenience of channel estimation, let's make the last carriers also be a pilot
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
P = P+1

# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers)

print ("allCarriers:   %s" % allCarriers)
print ("pilotCarriers: %s" % pilotCarriers)
print ("dataCarriers:  %s" % dataCarriers)
plt.figure(figsize=(8,0.8))
plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
plt.legend(fontsize=10, ncol=2)
plt.xlim((-1,K)); plt.ylim((-0.1, 0.3))
plt.xlabel('Carrier index')
plt.yticks([])
plt.grid(True);

```

    allCarriers:   [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
     50 51 52 53 54 55 56 57 58 59 60 61 62 63]
    pilotCarriers: [ 0  8 16 24 32 40 48 56 63]
    dataCarriers:  [ 1  2  3  4  5  6  7  9 10 11 12 13 14 15 17 18 19 20 21 22 23 25 26 27 28
     29 30 31 33 34 35 36 37 38 39 41 42 43 44 45 46 47 49 50 51 52 53 54 55 57
     58 59 60 61 62]



![png](/posts/4-ofdm-python_files/4-ofdm-python_10_1.png)


Let's define the modulation index $\mu$ and the corresponding mapping table. We consider 16QAM transmission, i.e. we have $\mu=4$ bits per symbol. Furthermore, the mapping from groups of 4 bits to a 16QAM constellation symbol shall be defined in `mapping_table`.


```python
mu = 4 # bits per symbol (i.e. 16QAM)
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}
for b3 in [0, 1]:
    for b2 in [0, 1]:
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = (b3, b2, b1, b0)
                Q = mapping_table[B]
                plt.plot(Q.real, Q.imag, 'bo')
                plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')
plt.grid(True)
plt.xlim((-4, 4)); plt.ylim((-4,4)); plt.xlabel('Real part (I)'); plt.ylabel('Imaginary part (Q)')
plt.title('16 QAM Constellation with Gray-Mapping');
```


![png]/posts/4-ofdm-python_files/4-ofdm-python_12_0.png)


Above, we have plotted the 16QAM constellation, along with the bit-labels. Note the Gray-mapping, i.e. two adjacent constellation symbols differ only by one bit and the other 3 bits remain the same. This technique helps to minimize bit-errors, in case a wrong constellation symbol is detected: Most probably, symbol errors are "off-by-one" errors, i.e. a symbol next to the correct symbol is detected. Then, only a single bit-error occurs.

The demapping table is simply the inverse mapping of the mapping table:


```python
demapping_table = {v : k for k, v in mapping_table.items()}
```

Let us now define the wireless channel between transmitter and receiver. Here, we use a two-tap multipath channel with given impulse response `channelResponse`. Also, we plot the corresponding frequency response. As we see, the channel is frequency-selective. Further, we define the signal-to-noise ratio in dB, that should occur at the receiver.


```python
channelResponse = np.array([1, 0, 0.3+0.3j])  # the impulse response of the wireless channel
H_exact = np.fft.fft(channelResponse, K)
plt.plot(allCarriers, abs(H_exact))
plt.xlabel('Subcarrier index'); plt.ylabel('$|H(f)|$'); plt.grid(True); plt.xlim(0, K-1)

SNRdb = 25  # signal to noise-ratio in dB at the receiver 
```


![png](/posts/4-ofdm-python_files/4-ofdm-python_16_0.png)


Now, that we have defined the necessary parameters for our OFDM example, let us consider the blocks in the OFDM system. Reconsider the block diagram:


<!-- ```python
%%tikz -l positioning,arrows --size 800,300
\tikzset{block/.style={draw,thick,minimum width=1cm,minimum height=1cm,align=center}}
\tikzset{node distance=0.5cm}
\tikzset{double distance=1pt}
\tikzset{>=latex}

\begin{scope}
\draw [->] (-0.5,0) node [left] {$\vec{b}$}  -- (0,0) node (SP1) [right,block] {S/P}; 
\node (M) [block,right=of SP1] {Mapping};
\node (IDFT) [block,right=of M] {IDFT};
\node (PS1) [block,right=of IDFT] {P/S};
\node (CP) [block,right=of PS1] {Add CP};

\draw [double,->] (PS1) -- (CP);

\def\lines{
\draw \ls ([yshift=3.mm]\from.east) -- ([yshift=3.mm]\to.west); 
\draw \ls ([yshift=1.mm]\from.east) -- ([yshift=1.mm]\to.west); 
\draw \ls ([yshift=-1.mm]\from.east) -- ([yshift=-1.mm]\to.west); 
\draw \ls ([yshift=-3.mm]\from.east) -- ([yshift=-3.mm]\to.west);   
}

\def\ls{[->]}
\def\from{SP1} \def\to{M} \lines;
\def\ls{[->,double]}
\def\from{M} \def\to{IDFT} \lines;
\def\from{IDFT} \def\to{PS1} \lines;



\node (C) [block,below right=of CP] {Channel}; 

\node (CP1) [block,below left=of C] {Remove CP};
\node (SP2) [block,left=of CP1] {S/P};
\node (DFT) [block,left=of SP2] {DFT};
\node (EQ) [block,left=of DFT] {Equalize};
\node (CE) [block,below=of EQ] {Channel\\Estimate};
\node (Dem) [block,left=of EQ] {Demapping};
\node (PS2) [block,left=of Dem] {P/S};
\draw [->] (PS2.west) -- +(-0.5,0) node [left] {$\hat{b}$};

\def\ls{[<-,double]}
\def\from{SP2} \def\to{CP1} \lines;
\def\from{DFT} \def\to{SP2} \lines;
\def\from{EQ} \def\to{DFT} \lines;
\def\from{Dem} \def\to{EQ} \lines;
\def\ls{[<-]}
\def\from{PS2} \def\to{Dem} \lines;

\draw [->,double,thick] (DFT.south) |- (CE.east);
\draw [->,double,thick] (CE.north) -- (EQ.south);

\draw [->,double] (CP) -| (C);
\draw [->,double] (C) |- (CP1);

\end{scope}
``` -->


![png](/posts/4-ofdm-python_files/4-ofdm-python_18_0.png)


It all starts with a random bit sequence $b$. We generate the according bits by a random generator that draws from a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution "Wikipedia link") with $p=0.5$, i.e. 1 and 0 have equal probability. Note, that the Bernoulli distribution is a special case of the [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution "Wikipedia link"), when only one draw is considered ($n=1$):


```python
bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
print ("Bits count: ", len(bits))
print ("First 20 bits: ", bits[:20])
print ("Mean of bits (should be around 0.5): ", np.mean(bits))
```

    Bits count:  220
    First 20 bits:  [0 0 1 0 1 0 0 1 1 1 0 1 0 1 1 1 1 1 1 0]
    Mean of bits (should be around 0.5):  0.518181818182


The `bits` are now sent to a serial-to-parallel converter, which groups the bits for the OFDM frame into a groups of $mu$ bits (i.e. one group for each subcarrier):


```python
def SP(bits):
    return bits.reshape((len(dataCarriers), mu))
bits_SP = SP(bits)
print ("First 5 bit groups")
print (bits_SP[:5,:])
```

    First 5 bit groups
    [[0 0 1 0]
     [1 0 0 1]
     [1 1 0 1]
     [0 1 1 1]
     [1 1 1 0]]


Now, the bits groups are sent to the mapper. The mapper converts the groups into complex-valued constellation symbols according to the `mapping_table`.


```python
def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])
QAM = Mapping(bits_SP)
print ("First 5 QAM symbols and bits:")
print (bits_SP[:5,:])
print (QAM[:5])
```

    First 5 QAM symbols and bits:
    [[0 0 1 0]
     [1 0 0 1]
     [1 1 0 1]
     [0 1 1 1]
     [1 1 1 0]]
    [-3.+3.j  3.-1.j  1.-1.j -1.+1.j  1.+3.j]


The next step (which is not shown in the diagram) is the allocation of different subcarriers with data and pilots. For each subcarrier we have defined wether it carries data or a pilot by the arrays `dataCarriers` and `pilotCarriers`. Now, to create the overall OFDM data, we need to put the data and pilots into the OFDM carriers:


```python
def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol
OFDM_data = OFDM_symbol(QAM)
print ("Number of OFDM carriers in frequency domain: ", len(OFDM_data))
```

    Number of OFDM carriers in frequency domain:  64


Now, the OFDM carriers contained in `OFDM_data` can be transformed to the time-domain by means of the IDFT operation. 


```python
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)
OFDM_time = IDFT(OFDM_data)
print ("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))
```

    Number of OFDM samples in time-domain before CP:  64


Subsequently, we add a cyclic prefix to the symbol. This operation concatenates a copy of the last `CP` samples of the OFDM time domain signal to the beginning. This way, a cyclic extension is achieved. The CP fulfills two tasks:

1. It isolates different OFDM blocks from each other when the wireless channel contains multiple paths, i.e. is frequency-selective.
2. It turns the linear convolution with the channel into a circular one. Only with a circular convolution, we can use the single-tap equalization OFDM is so famous for. 


```python
def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning
OFDM_withCP = addCP(OFDM_time)
print ("Number of OFDM samples in time domain with CP: ", len(OFDM_withCP))
```

    Number of OFDM samples in time domain with CP:  80


Now, the signal is sent to the antenna and sent over the air to the receiver. In between both antennas, there is the wireless channel. We model this channel as a static multipath channel with impulse response `channelResponse`. Hence, the signal at the receive antenna is the convolution of the transmit signal with the channel response. Additionally, we add some noise to the signal according to the given SNR value:


```python
def channel(signal):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
    
    print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
    
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise
OFDM_TX = OFDM_withCP
OFDM_RX = channel(OFDM_TX)
plt.figure(figsize=(8,2))
plt.plot(abs(OFDM_TX), label='TX signal')
plt.plot(abs(OFDM_RX), label='RX signal')
plt.legend(fontsize=10)
plt.xlabel('Time'); plt.ylabel('$|x(t)|$');
plt.grid(True);
```

    RX Signal power: 0.2114. Noise power: 0.0007



![png](/posts/4-ofdm-python_files/4-ofdm-python_32_1.png)


Now, at the receiver the CP is removed from the signal and a window of $K$ samples is extracted from the received signal.


```python
def removeCP(signal):
    return signal[CP:(CP+K)]
OFDM_RX_noCP = removeCP(OFDM_RX)
```

Afterwards, the signal is transformed back to the frequency domain, in order to have the received value on each subcarrier available.


```python
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)
OFDM_demod = DFT(OFDM_RX_noCP)
```

As the next step, the wireless channel needs to be estimated. For illustration purposes, we resort to a simple zero-forcing channel estimation followed by a simple interpolation. The principle of channel estimation is as follows:

The transmit signal contains pilot values at certain pilot carriers. These pilot values and their position in the frequency domain (i.e. the pilot carrier index) are known to the receiver. From the received information at the pilot subcarriers, the receiver can estimate the effect of the wireless channel onto this subcarrier (because it knows what was transmitted and what was received). Hence, the receiver gains information about the wireless channel at the pilot carriers. However, it wants to know what happened at the data carriers. To achieve this, it interpolates the channel values between the pilot carriers to get an estimate of the channel in the data carriers.


```python
def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
    
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    
    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
    plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    plt.ylim(0,2)
    
    return Hest
Hest = channelEstimate(OFDM_demod)
```


![png](/posts/4-ofdm-python_files/4-ofdm-python_38_0.png)


Now that the channel is estimated at all carriers, we can use this information in the channel equalizer step. Here, for each subcarrier, the influence of the channel is removed such that we get the clear (only noisy) constellation symbols back.


```python
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest
equalized_Hest = equalize(OFDM_demod, Hest)
```

The next step (not shown in the diagram) is to extract the data carriers from the equalized symbol. Here, we throw away the pilot carriers, as they do not provide any information, but were used for the channel estimation process.


```python
def get_payload(equalized):
    return equalized[dataCarriers]
QAM_est = get_payload(equalized_Hest)
plt.plot(QAM_est.real, QAM_est.imag, 'bo');
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary Part'); plt.title("Received constellation");
```


![png](/posts/4-ofdm-python_files/4-ofdm-python_42_0.png)


Now, that the constellation is obtained back, we need to send the complex values to the demapper, to transform the constellation points to the bit groups. In order to do this, we compare each received constellation point against each possible constellation point and choose the constellation point which is closest to the received point. Then, we return the bit-group that belongs to this point.


```python
def Demapping(QAM):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

PS_est, hardDecision = Demapping(QAM_est)
for qam, hard in zip(QAM_est, hardDecision):
    plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
    plt.plot(hardDecision.real, hardDecision.imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Hard Decision demapping');
```


![png](/posts/4-ofdm-python_files/4-ofdm-python_44_0.png)


In the diagram above, the blue points are the received QAM points, where as the the red points connected to them are the closest possible constellation points, and the bit groups corresponding to these red points are returned. 

Finally, the bit groups need to be converted to a serial stream of bits, by means of parallel to serial conversion.


```python
def PS(bits):
    return bits.reshape((-1,))
bits_est = PS(PS_est)
```

Now, that all bits are decoded, let's calculate the bit error rate:


```python
print ("Obtained Bit error rate: ", np.sum(abs(bits-bits_est))/len(bits))
```

    Obtained Bit error rate:  0.0


## Exercise 1

Tinker with the SNR to observe some bit error.

## References

1. MIMO OFDM Wireless Communications, Y. Chang and W. Yang, IEEE Press, 2010.
2. [Sympy Docs](https://www.sympy.org).
3. [Numpy Docs](https://numpy.org).
