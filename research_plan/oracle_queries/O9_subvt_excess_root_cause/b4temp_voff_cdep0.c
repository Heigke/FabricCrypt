                      return(E_BADPARM);
                  }


                  if (model->BSIM4binUnit == 1)
                  {   Inv_L = 1.0e-6 / pParam->BSIM4leff;
                      Inv_W = 1.0e-6 / pParam->BSIM4weff;
                      Inv_LW = 1.0e-12 / (pParam->BSIM4leff
                             * pParam->BSIM4weff);
                  }
                  else
                  {   Inv_L = 1.0 / pParam->BSIM4leff;
                      Inv_W = 1.0 / pParam->BSIM4weff;
                      Inv_LW = 1.0 / (pParam->BSIM4leff
                             * pParam->BSIM4weff);
                  }
                  pParam->BSIM4cdsc = model->BSIM4cdsc
                                    + model->BSIM4lcdsc * Inv_L
                                    + model->BSIM4wcdsc * Inv_W
                                    + model->BSIM4pcdsc * Inv_LW;
                  pParam->BSIM4cdscb = model->BSIM4cdscb
                                     + model->BSIM4lcdscb * Inv_L
                                     + model->BSIM4wcdscb * Inv_W
                                     + model->BSIM4pcdscb * Inv_LW;

                      pParam->BSIM4cdscd = model->BSIM4cdscd
                                     + model->BSIM4lcdscd * Inv_L
                                     + model->BSIM4wcdscd * Inv_W
                                     + model->BSIM4pcdscd * Inv_LW;

                  pParam->BSIM4cit = model->BSIM4cit
                                   + model->BSIM4lcit * Inv_L
                                   + model->BSIM4wcit * Inv_W
                                   + model->BSIM4pcit * Inv_LW;
                  pParam->BSIM4nfactor = model->BSIM4nfactor
                                       + model->BSIM4lnfactor * Inv_L
                                       + model->BSIM4wnfactor * Inv_W
                                       + model->BSIM4pnfactor * Inv_LW;
                  pParam->BSIM4tnfactor = model->BSIM4tnfactor                        /* v4.7 */
                                       + model->BSIM4ltnfactor * Inv_L
                                       + model->BSIM4wtnfactor * Inv_W
                                       + model->BSIM4ptnfactor * Inv_LW;
                  pParam->BSIM4xj = model->BSIM4xj
                                  + model->BSIM4lxj * Inv_L
                                  + model->BSIM4wxj * Inv_W
                                  + model->BSIM4pxj * Inv_LW;
                  pParam->BSIM4vsat = model->BSIM4vsat
                                    + model->BSIM4lvsat * Inv_L
                                    + model->BSIM4wvsat * Inv_W
                                    + model->BSIM4pvsat * Inv_LW;
                  pParam->BSIM4at = model->BSIM4at
                                  + model->BSIM4lat * Inv_L
                                  + model->BSIM4wat * Inv_W
                                  + model->BSIM4pat * Inv_LW;
                  pParam->BSIM4a0 = model->BSIM4a0
                                  + model->BSIM4la0 * Inv_L
                                  + model->BSIM4wa0 * Inv_W
                                  + model->BSIM4pa0 * Inv_LW;

                  pParam->BSIM4ags = model->BSIM4ags
                                  + model->BSIM4lags * Inv_L
                                  + model->BSIM4wags * Inv_W
                                  + model->BSIM4pags * Inv_LW;

                  pParam->BSIM4a1 = model->BSIM4a1
                                  + model->BSIM4la1 * Inv_L
                                  + model->BSIM4wa1 * Inv_W
                                  + model->BSIM4pa1 * Inv_LW;
                  pParam->BSIM4a2 = model->BSIM4a2
                                  + model->BSIM4la2 * Inv_L
                                  + model->BSIM4wa2 * Inv_W
                                  + model->BSIM4pa2 * Inv_LW;
                  pParam->BSIM4keta = model->BSIM4keta
                                    + model->BSIM4lketa * Inv_L
                                    + model->BSIM4wketa * Inv_W
                                    + model->BSIM4pketa * Inv_LW;
                  pParam->BSIM4nsub = model->BSIM4nsub
                                    + model->BSIM4lnsub * Inv_L
                                    + model->BSIM4wnsub * Inv_W
                                    + model->BSIM4pnsub * Inv_LW;
                  pParam->BSIM4ndep = model->BSIM4ndep
                                    + model->BSIM4lndep * Inv_L
                                    + model->BSIM4wndep * Inv_W
                                    + model->BSIM4pndep * Inv_LW;
                  pParam->BSIM4nsd = model->BSIM4nsd
                                   + model->BSIM4lnsd * Inv_L
                                   + model->BSIM4wnsd * Inv_W
                                   + model->BSIM4pnsd * Inv_LW;
                  pParam->BSIM4phin = model->BSIM4phin
                                    + model->BSIM4lphin * Inv_L
                                    + model->BSIM4wphin * Inv_W
                                    + model->BSIM4pphin * Inv_LW;
                  pParam->BSIM4ngate = model->BSIM4ngate
                                     + model->BSIM4lngate * Inv_L
                                     + model->BSIM4wngate * Inv_W
                                     + model->BSIM4pngate * Inv_LW;
                  pParam->BSIM4gamma1 = model->BSIM4gamma1
                                      + model->BSIM4lgamma1 * Inv_L
                                      + model->BSIM4wgamma1 * Inv_W
                                      + model->BSIM4pgamma1 * Inv_LW;
                  pParam->BSIM4gamma2 = model->BSIM4gamma2
                                      + model->BSIM4lgamma2 * Inv_L
                                      + model->BSIM4wgamma2 * Inv_W
                                      + model->BSIM4pgamma2 * Inv_LW;
                  pParam->BSIM4vbx = model->BSIM4vbx
                                   + model->BSIM4lvbx * Inv_L
                                   + model->BSIM4wvbx * Inv_W
                                   + model->BSIM4pvbx * Inv_LW;
                  pParam->BSIM4vbm = model->BSIM4vbm
                                   + model->BSIM4lvbm * Inv_L
                                   + model->BSIM4wvbm * Inv_W
                                   + model->BSIM4pvbm * Inv_LW;
                  pParam->BSIM4xt = model->BSIM4xt
                                   + model->BSIM4lxt * Inv_L
                                   + model->BSIM4wxt * Inv_W
                                   + model->BSIM4pxt * Inv_LW;
                  pParam->BSIM4vfb = model->BSIM4vfb
                                   + model->BSIM4lvfb * Inv_L
                                   + model->BSIM4wvfb * Inv_W
                                   + model->BSIM4pvfb * Inv_LW;
                  pParam->BSIM4k1 = model->BSIM4k1
                                  + model->BSIM4lk1 * Inv_L
                                  + model->BSIM4wk1 * Inv_W
                                  + model->BSIM4pk1 * Inv_LW;
                  pParam->BSIM4kt1 = model->BSIM4kt1
                                   + model->BSIM4lkt1 * Inv_L
                                   + model->BSIM4wkt1 * Inv_W
                                   + model->BSIM4pkt1 * Inv_LW;
                  pParam->BSIM4kt1l = model->BSIM4kt1l
                                    + model->BSIM4lkt1l * Inv_L
                                    + model->BSIM4wkt1l * Inv_W
                                    + model->BSIM4pkt1l * Inv_LW;
                  pParam->BSIM4k2 = model->BSIM4k2
                                  + model->BSIM4lk2 * Inv_L
                                  + model->BSIM4wk2 * Inv_W
                                  + model->BSIM4pk2 * Inv_LW;
                  pParam->BSIM4kt2 = model->BSIM4kt2
                                   + model->BSIM4lkt2 * Inv_L
                                   + model->BSIM4wkt2 * Inv_W
                                   + model->BSIM4pkt2 * Inv_LW;
                  pParam->BSIM4k3 = model->BSIM4k3
                                  + model->BSIM4lk3 * Inv_L
                                  + model->BSIM4wk3 * Inv_W
                                  + model->BSIM4pk3 * Inv_LW;
                  pParam->BSIM4k3b = model->BSIM4k3b
                                   + model->BSIM4lk3b * Inv_L
                                   + model->BSIM4wk3b * Inv_W
                                   + model->BSIM4pk3b * Inv_LW;
                  pParam->BSIM4w0 = model->BSIM4w0
                                  + model->BSIM4lw0 * Inv_L
                                  + model->BSIM4ww0 * Inv_W
                                  + model->BSIM4pw0 * Inv_LW;
                  pParam->BSIM4lpe0 = model->BSIM4lpe0
                                    + model->BSIM4llpe0 * Inv_L
                                     + model->BSIM4wlpe0 * Inv_W
                                    + model->BSIM4plpe0 * Inv_LW;
                  pParam->BSIM4lpeb = model->BSIM4lpeb
                                    + model->BSIM4llpeb * Inv_L
                                    + model->BSIM4wlpeb * Inv_W
                                    + model->BSIM4plpeb * Inv_LW;
                  pParam->BSIM4dvtp0 = model->BSIM4dvtp0
                                     + model->BSIM4ldvtp0 * Inv_L
                                     + model->BSIM4wdvtp0 * Inv_W
                                     + model->BSIM4pdvtp0 * Inv_LW;
                  pParam->BSIM4dvtp1 = model->BSIM4dvtp1
                                     + model->BSIM4ldvtp1 * Inv_L
                                     + model->BSIM4wdvtp1 * Inv_W
                                     + model->BSIM4pdvtp1 * Inv_LW;
                  pParam->BSIM4dvtp2 = model->BSIM4dvtp2                 /* v4.7  */
                                     + model->BSIM4ldvtp2 * Inv_L
                                     + model->BSIM4wdvtp2 * Inv_W
                                     + model->BSIM4pdvtp2 * Inv_LW;
                  pParam->BSIM4dvtp3 = model->BSIM4dvtp3                 /* v4.7  */
                                     + model->BSIM4ldvtp3 * Inv_L
                                     + model->BSIM4wdvtp3 * Inv_W
                                     + model->BSIM4pdvtp3 * Inv_LW;
                  pParam->BSIM4dvtp4 = model->BSIM4dvtp4                 /* v4.7  */
                                     + model->BSIM4ldvtp4 * Inv_L
                                     + model->BSIM4wdvtp4 * Inv_W
                                     + model->BSIM4pdvtp4 * Inv_LW;
                  pParam->BSIM4dvtp5 = model->BSIM4dvtp5                 /* v4.7  */
                                     + model->BSIM4ldvtp5 * Inv_L
                                     + model->BSIM4wdvtp5 * Inv_W
                                     + model->BSIM4pdvtp5 * Inv_LW;
                  pParam->BSIM4dvt0 = model->BSIM4dvt0
                                    + model->BSIM4ldvt0 * Inv_L
                                    + model->BSIM4wdvt0 * Inv_W
                                    + model->BSIM4pdvt0 * Inv_LW;
                  pParam->BSIM4dvt1 = model->BSIM4dvt1
                                    + model->BSIM4ldvt1 * Inv_L
                                    + model->BSIM4wdvt1 * Inv_W
                                    + model->BSIM4pdvt1 * Inv_LW;
                  pParam->BSIM4dvt2 = model->BSIM4dvt2
                                    + model->BSIM4ldvt2 * Inv_L
                                    + model->BSIM4wdvt2 * Inv_W
                                    + model->BSIM4pdvt2 * Inv_LW;
                  pParam->BSIM4dvt0w = model->BSIM4dvt0w
                                    + model->BSIM4ldvt0w * Inv_L
                                    + model->BSIM4wdvt0w * Inv_W
                                    + model->BSIM4pdvt0w * Inv_LW;
                  pParam->BSIM4dvt1w = model->BSIM4dvt1w
                                    + model->BSIM4ldvt1w * Inv_L
                                    + model->BSIM4wdvt1w * Inv_W
                                    + model->BSIM4pdvt1w * Inv_LW;
                  pParam->BSIM4dvt2w = model->BSIM4dvt2w
                                    + model->BSIM4ldvt2w * Inv_L
                                    + model->BSIM4wdvt2w * Inv_W
                                    + model->BSIM4pdvt2w * Inv_LW;
                  pParam->BSIM4drout = model->BSIM4drout
                                     + model->BSIM4ldrout * Inv_L
                                     + model->BSIM4wdrout * Inv_W
                                     + model->BSIM4pdrout * Inv_LW;
                  pParam->BSIM4dsub = model->BSIM4dsub
                                    + model->BSIM4ldsub * Inv_L
                                    + model->BSIM4wdsub * Inv_W
                                    + model->BSIM4pdsub * Inv_LW;
                  pParam->BSIM4vth0 = model->BSIM4vth0
                                    + model->BSIM4lvth0 * Inv_L
                                    + model->BSIM4wvth0 * Inv_W
                                    + model->BSIM4pvth0 * Inv_LW;
                  pParam->BSIM4ua = model->BSIM4ua
                                  + model->BSIM4lua * Inv_L
                                  + model->BSIM4wua * Inv_W
                                  + model->BSIM4pua * Inv_LW;
                  pParam->BSIM4ua1 = model->BSIM4ua1
                                   + model->BSIM4lua1 * Inv_L
                                   + model->BSIM4wua1 * Inv_W
                                   + model->BSIM4pua1 * Inv_LW;
                  pParam->BSIM4ub = model->BSIM4ub
                                  + model->BSIM4lub * Inv_L
                                  + model->BSIM4wub * Inv_W
                                  + model->BSIM4pub * Inv_LW;
                  pParam->BSIM4ub1 = model->BSIM4ub1
                                   + model->BSIM4lub1 * Inv_L
                                   + model->BSIM4wub1 * Inv_W
                                   + model->BSIM4pub1 * Inv_LW;
                  pParam->BSIM4uc = model->BSIM4uc
                                  + model->BSIM4luc * Inv_L
                                  + model->BSIM4wuc * Inv_W
                                  + model->BSIM4puc * Inv_LW;
                  pParam->BSIM4uc1 = model->BSIM4uc1
                                   + model->BSIM4luc1 * Inv_L
                                   + model->BSIM4wuc1 * Inv_W
                                   + model->BSIM4puc1 * Inv_LW;
                  pParam->BSIM4ud = model->BSIM4ud
                                  + model->BSIM4lud * Inv_L
                                  + model->BSIM4wud * Inv_W
                                  + model->BSIM4pud * Inv_LW;
                  pParam->BSIM4ud1 = model->BSIM4ud1
                                  + model->BSIM4lud1 * Inv_L
                                  + model->BSIM4wud1 * Inv_W
                                  + model->BSIM4pud1 * Inv_LW;
                  pParam->BSIM4up = model->BSIM4up
                                  + model->BSIM4lup * Inv_L
                                  + model->BSIM4wup * Inv_W
                                  + model->BSIM4pup * Inv_LW;
                  pParam->BSIM4lp = model->BSIM4lp
                                  + model->BSIM4llp * Inv_L
                                  + model->BSIM4wlp * Inv_W
                                  + model->BSIM4plp * Inv_LW;
                  pParam->BSIM4eu = model->BSIM4eu
                                  + model->BSIM4leu * Inv_L
                                  + model->BSIM4weu * Inv_W
                                  + model->BSIM4peu * Inv_LW;
                  pParam->BSIM4u0 = model->BSIM4u0
                                  + model->BSIM4lu0 * Inv_L
                                  + model->BSIM4wu0 * Inv_W
                                  + model->BSIM4pu0 * Inv_LW;
                  pParam->BSIM4ute = model->BSIM4ute
                                   + model->BSIM4lute * Inv_L
                                   + model->BSIM4wute * Inv_W
                                   + model->BSIM4pute * Inv_LW;
                /*high k mobility*/
                 pParam->BSIM4ucs = model->BSIM4ucs
                                  + model->BSIM4lucs * Inv_L
                                  + model->BSIM4wucs * Inv_W
                                  + model->BSIM4pucs * Inv_LW;
                  pParam->BSIM4ucste = model->BSIM4ucste
                           + model->BSIM4lucste * Inv_L
                                   + model->BSIM4wucste * Inv_W
                                   + model->BSIM4pucste * Inv_LW;

                  pParam->BSIM4voff = model->BSIM4voff
                                    + model->BSIM4lvoff * Inv_L
                                    + model->BSIM4wvoff * Inv_W
                                    + model->BSIM4pvoff * Inv_LW;
                  pParam->BSIM4tvoff = model->BSIM4tvoff
                                    + model->BSIM4ltvoff * Inv_L
                                    + model->BSIM4wtvoff * Inv_W
                                    + model->BSIM4ptvoff * Inv_LW;
                  pParam->BSIM4minv = model->BSIM4minv
                  pParam->BSIM4BechvbEdge = -pParam->BSIM4Bechvb
                                          * toxe * pParam->BSIM4poxedge;
                  pParam->BSIM4Aechvb *= pParam->BSIM4weff * pParam->BSIM4leff
                                       * pParam->BSIM4ToxRatio;
                  pParam->BSIM4Bechvb *= -toxe;


                  pParam->BSIM4mstar = 0.5 + atan(pParam->BSIM4minv) / PI;
                  pParam->BSIM4mstarcv = 0.5 + atan(pParam->BSIM4minvcv) / PI;
                  pParam->BSIM4voffcbn =  pParam->BSIM4voff + model->BSIM4voffl / pParam->BSIM4leff;
                  pParam->BSIM4voffcbncv =  pParam->BSIM4voffcv + model->BSIM4voffcvl / pParam->BSIM4leff;

                  pParam->BSIM4ldeb = sqrt(epssub * Vtm0 / (Charge_q
                                    * pParam->BSIM4ndep * 1.0e6)) / 3.0;
                  pParam->BSIM4acde *= pow((pParam->BSIM4ndep / 2.0e16), -0.25);

