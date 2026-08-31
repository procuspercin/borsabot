import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

INPUT = Path('data/base_30d_regime_predictions_enriched.csv')
WINDOW = 1260
TARGET_HORIZON = 30
MIN_HISTORY = 300
BASE_GATE_AUC = 0.53
BASE_GATE_BAL = 0.50
AVOID_AUC = 0.47
HIGH_AUC = 0.56

REGIME_COLS = ['MarketRegime','Momentum20Bucket','Momentum60Bucket','VolatilityBucket','SMA200Bucket']
BASE_LEVELS = [
    ('FULL',['MarketRegime','Momentum20Bucket','Momentum60Bucket','VolatilityBucket','SMA200Bucket']),
    ('TREND_VOL_M20',['MarketRegime','VolatilityBucket','Momentum20Bucket']),
    ('TREND_VOL',['MarketRegime','VolatilityBucket']),
    ('M20_VOL',['Momentum20Bucket','VolatilityBucket']),
    ('M20',['Momentum20Bucket']),
    ('MARKET',['MarketRegime']),
]
META_LEVELS = [
    ('M20_VOL',['Momentum20Bucket','VolatilityBucket']),
    ('MARKET_M20',['MarketRegime','Momentum20Bucket']),
    ('M20',['Momentum20Bucket']),
    ('M60',['Momentum60Bucket']),
]

def safe_auc(y,p):
    y=np.asarray(y); p=np.asarray(p)
    return np.nan if len(y)<2 or len(np.unique(y))<2 else roc_auc_score(y,p)

def calc_metrics(g):
    if len(g)==0:
        return {'Samples':0,'Accuracy':np.nan,'BalancedAccuracy':np.nan,'AUC':np.nan}
    y=g['actual'].to_numpy(); p=g['probability'].to_numpy(); pred=(p>=0.5).astype(int)
    bal=balanced_accuracy_score(y,pred) if len(np.unique(y))>=2 else np.nan
    return {'Samples':len(g),'Accuracy':np.mean(pred==y),'BalancedAccuracy':bal,'AUC':safe_auc(y,p)}

def build_group_map(hist, cols):
    out={}
    for key,g in hist.groupby(cols, observed=True, sort=False):
        if len(g)<MIN_HISTORY: continue
        if not isinstance(key,tuple): key=(key,)
        y=g['actual'].to_numpy(); p=g['probability'].to_numpy(); pred=(p>=0.5).astype(int)
        auc=safe_auc(y,p)
        if pd.isna(auc): continue
        bal=balanced_accuracy_score(y,pred) if len(np.unique(y))>=2 else np.nan
        out[tuple(str(x) for x in key)]={'samples':len(g),'auc':auc,'bal':bal}
    return out

def main():
    print('='*110)
    print('30D BASE — LEAKAGE-SAFE HIGH / NORMAL / AVOID META-GATE')
    print('='*110)
    print(f'Window={WINDOW} | MinHist={MIN_HISTORY} | BaseGate AUC>={BASE_GATE_AUC:.2f} | AVOID<{AVOID_AUC:.2f} | HIGH>={HIGH_AUC:.2f}')
    t0=time.time()
    df=pd.read_csv(INPUT, parse_dates=['Date'], low_memory=False).sort_values(['Date','Ticker']).reset_index(drop=True)
    for c in REGIME_COLS: df[c]=df[c].astype('string').fillna('NA')
    dates=df['Date'].drop_duplicates().sort_values().reset_index(drop=True)
    date_map={pd.Timestamp(d):i for i,d in enumerate(dates)}
    df['_date_idx']=df['Date'].map(date_map)
    df['_label_available_idx']=df['_date_idx']+TARGET_HORIZON
    df['Year']=df['Date'].dt.year
    by_date={int(di):g.index.to_numpy() for di,g in df.groupby('_date_idx',sort=True)}
    records=[]
    for di in range(len(dates)):
        today_ids=by_date.get(di)
        if today_ids is None: continue
        hist_start=max(0,di-WINDOW)
        hist=df[(df['_label_available_idx']<=di)&(df['_date_idx']>=hist_start)]
        base_maps={}; meta_maps={}
        if len(hist)>=MIN_HISTORY:
            for name,cols in BASE_LEVELS: base_maps[name]=(cols,build_group_map(hist,cols))
            for name,cols in META_LEVELS: meta_maps[name]=(cols,build_group_map(hist,cols))
        today=df.loc[today_ids]
        for rid,row in today.iterrows():
            gate_level='NONE'; gate_hist_samples=0; gate_hist_auc=np.nan; gate_hist_bal=np.nan
            for name,cols in BASE_LEVELS:
                if name not in base_maps: continue
                _,gmap=base_maps[name]; key=tuple(str(row[c]) for c in cols)
                if key in gmap:
                    m=gmap[key]; gate_level=name; gate_hist_samples=m['samples']; gate_hist_auc=m['auc']; gate_hist_bal=m['bal']; break
            base_gate_active=(pd.notna(gate_hist_auc) and gate_hist_auc>=BASE_GATE_AUC and pd.notna(gate_hist_bal) and gate_hist_bal>=BASE_GATE_BAL)
            meta_stats={}
            for name,cols in META_LEVELS:
                stat={'samples':0,'auc':np.nan,'bal':np.nan}
                if name in meta_maps:
                    _,gmap=meta_maps[name]; key=tuple(str(row[c]) for c in cols)
                    if key in gmap: stat=gmap[key]
                meta_stats[name]=stat
            bad=[]
            for name in ['MARKET_M20','M20','M60']:
                s=meta_stats[name]
                if s['samples']>=MIN_HISTORY and pd.notna(s['auc']) and s['auc']<AVOID_AUC: bad.append(name)
            avoid=len(bad)>0
            hs=meta_stats['M20_VOL']
            high=(base_gate_active and not avoid and hs['samples']>=MIN_HISTORY and pd.notna(hs['auc']) and hs['auc']>=HIGH_AUC)
            confidence='OFF' if not base_gate_active else ('AVOID' if avoid else ('HIGH' if high else 'NORMAL'))
            records.append({'_row_id':rid,'Confidence':confidence,'BaseGateActive':base_gate_active,'GateLevel':gate_level,'GateHistSamples':gate_hist_samples,'GateHistAUC':gate_hist_auc,'GateHistBalAcc':gate_hist_bal,'AvoidReason':','.join(bad),'M20VolHistSamples':hs['samples'],'M20VolHistAUC':hs['auc'],'MarketM20HistSamples':meta_stats['MARKET_M20']['samples'],'MarketM20HistAUC':meta_stats['MARKET_M20']['auc'],'M20HistSamples':meta_stats['M20']['samples'],'M20HistAUC':meta_stats['M20']['auc'],'M60HistSamples':meta_stats['M60']['samples'],'M60HistAUC':meta_stats['M60']['auc']})
        if di%300==0 or di==len(dates)-1: print(f'{di+1}/{len(dates)} | {pd.Timestamp(dates.iloc[di]).date()}')
    decisions=pd.DataFrame(records).set_index('_row_id').sort_index()
    out=df.join(decisions)

    strategies=[
        ('BASE_ALL', pd.Series(True,index=out.index)),
        ('BASE_GATE_1260_053', out['BaseGateActive']==True),
        ('META_HIGH_NORMAL', out['Confidence'].isin(['HIGH','NORMAL'])),
        ('HIGH_ONLY', out['Confidence']=='HIGH'),
        ('NORMAL_ONLY', out['Confidence']=='NORMAL'),
        ('AVOID_ONLY', out['Confidence']=='AVOID'),
    ]
    summary=[]
    for name,mask in strategies:
        m=calc_metrics(out[mask]); m['Strategy']=name; m['Coverage']=mask.mean(); summary.append(m)
    summary=pd.DataFrame(summary)[['Strategy','Samples','Coverage','Accuracy','BalancedAccuracy','AUC']]
    print('\n'+'='*110+'\nOVERALL\n'+'='*110)
    p=summary.copy(); p['Coverage']*=100; p['Accuracy']*=100; p['BalancedAccuracy']*=100
    print(p.to_string(index=False,formatters={'Coverage':lambda x:f'%{x:.2f}','Accuracy':lambda x:f'%{x:.2f}','BalancedAccuracy':lambda x:'nan' if pd.isna(x) else f'%{x:.2f}','AUC':lambda x:'nan' if pd.isna(x) else f'{x:.4f}'}))

    yearly=[]
    for year in sorted(out['Year'].unique()):
        y=out[out['Year']==year]
        by=calc_metrics(y); gy=calc_metrics(y[y['BaseGateActive']==True]); my=calc_metrics(y[y['Confidence'].isin(['HIGH','NORMAL'])]); hy=calc_metrics(y[y['Confidence']=='HIGH']); ay=calc_metrics(y[y['Confidence']=='AVOID'])
        yearly.append({'Year':year,'BaseAUC':by['AUC'],'BaseGateAUC':gy['AUC'],'MetaAUC':my['AUC'],'DeltaMetaVsBase':my['AUC']-by['AUC'] if pd.notna(my['AUC']) and pd.notna(by['AUC']) else np.nan,'DeltaMetaVsBaseGate':my['AUC']-gy['AUC'] if pd.notna(my['AUC']) and pd.notna(gy['AUC']) else np.nan,'MetaBalAcc':my['BalancedAccuracy'],'MetaCoverage':len(y[y['Confidence'].isin(['HIGH','NORMAL'])])/len(y) if len(y) else np.nan,'HighSamples':hy['Samples'],'HighAUC':hy['AUC'],'AvoidSamples':ay['Samples'],'AvoidAUC':ay['AUC']})
    yearly=pd.DataFrame(yearly)
    print('\n'+'='*110+'\nYEAR BY YEAR\n'+'='*110)
    py=yearly.copy(); py['MetaCoverage']*=100; py['MetaBalAcc']*=100
    print(py.to_string(index=False,formatters={'BaseAUC':lambda x:'nan' if pd.isna(x) else f'{x:.4f}','BaseGateAUC':lambda x:'nan' if pd.isna(x) else f'{x:.4f}','MetaAUC':lambda x:'nan' if pd.isna(x) else f'{x:.4f}','DeltaMetaVsBase':lambda x:'nan' if pd.isna(x) else f'{x:+.4f}','DeltaMetaVsBaseGate':lambda x:'nan' if pd.isna(x) else f'{x:+.4f}','MetaBalAcc':lambda x:'nan' if pd.isna(x) else f'%{x:.2f}','MetaCoverage':lambda x:f'%{x:.2f}','HighAUC':lambda x:'nan' if pd.isna(x) else f'{x:.4f}','AvoidAUC':lambda x:'nan' if pd.isna(x) else f'{x:.4f}'}))

    print('\n'+'='*110+'\nCONFIDENCE COUNTS\n'+'='*110)
    print(out['Confidence'].value_counts(dropna=False).to_string())
    avoid_reasons=out.loc[out['Confidence']=='AVOID','AvoidReason'].value_counts().rename_axis('Reason').reset_index(name='Samples')
    print('\n'+'='*110+'\nAVOID REASONS\n'+'='*110)
    print(avoid_reasons.to_string(index=False) if len(avoid_reasons) else 'No AVOID rows.')

    summary.to_csv('data/base_30d_meta_gate_summary.csv',index=False)
    yearly.to_csv('data/base_30d_meta_gate_yearly.csv',index=False)
    out.to_csv('data/base_30d_meta_gate_predictions.csv',index=False)
    avoid_reasons.to_csv('data/base_30d_meta_gate_avoid_reasons.csv',index=False)
    print('\nSaved: data/base_30d_meta_gate_summary.csv, data/base_30d_meta_gate_yearly.csv, data/base_30d_meta_gate_predictions.csv, data/base_30d_meta_gate_avoid_reasons.csv')
    print(f'Toplam süre: {(time.time()-t0)/60:.2f} dakika')
    print('NOT: VolatilityBucket mevcut global bucketlardan geliyor; label leakage yok ama bucket cutoffları henüz tam production-safe değil.')

if __name__=='__main__': main()
