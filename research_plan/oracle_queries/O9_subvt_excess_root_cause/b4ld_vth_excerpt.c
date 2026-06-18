          else
          {   T4 = 1.0 / (3.0 + 8.0 * T0);
              T1 = (1.0 + 3.0 * T0) * T4; 
              T2 = pParam->BSIM4dvt2w * T4 * T4;
          }
          ltw = model->BSIM4factor1 * T3 * T1;
          dltw_dVb = model->BSIM4factor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2);

          T0 = pParam->BSIM4dvt1 * Leff / lt1;
          if (T0 < EXP_THRESHOLD)
          {   T1 = exp(T0);
              T2 = T1 - 1.0;
              T3 = T2 * T2;
              T4 = T3 + 2.0 * T1 * MIN_EXP;
              Theta0 = T1 / T4;
              dT1_dVb = -T0 * T1 * dlt1_dVb / lt1;
              dTheta0_dVb = dT1_dVb * (T4 - 2.0 * T1 * (T2 + MIN_EXP)) / T4 / T4;
          }
          else
          {   Theta0 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
              dTheta0_dVb = 0.0;
          }
          here->BSIM4thetavth = pParam->BSIM4dvt0 * Theta0;
          Delt_vth = here->BSIM4thetavth * V0;
          dDelt_vth_dVb = pParam->BSIM4dvt0 * dTheta0_dVb * V0;

          T0 = pParam->BSIM4dvt1w * pParam->BSIM4weff * Leff / ltw;
          if (T0 < EXP_THRESHOLD)
          {   T1 = exp(T0);
              T2 = T1 - 1.0;
              T3 = T2 * T2;
              T4 = T3 + 2.0 * T1 * MIN_EXP;
              T5 = T1 / T4;
              dT1_dVb = -T0 * T1 * dltw_dVb / ltw; 
              dT5_dVb = dT1_dVb * (T4 - 2.0 * T1 * (T2 + MIN_EXP)) / T4 / T4;
          }
          else
          {   T5 = 1.0 / (MAX_EXP - 2.0); /* 3.0 * MIN_EXP omitted */
              dT5_dVb = 0.0;
          }
          T0 = pParam->BSIM4dvt0w * T5;
          T2 = T0 * V0;
          dT2_dVb = pParam->BSIM4dvt0w * dT5_dVb * V0;

          TempRatio =  ckt->CKTtemp / model->BSIM4tnom - 1.0;
          T0 = sqrt(1.0 + pParam->BSIM4lpe0 / Leff);
          T1 = pParam->BSIM4k1ox * (T0 - 1.0) * pParam->BSIM4sqrtPhi
             + (pParam->BSIM4kt1 + pParam->BSIM4kt1l / Leff
             + pParam->BSIM4kt2 * Vbseff) * TempRatio;
          Vth_NarrowW = toxe * pParam->BSIM4phi
                      / (pParam->BSIM4weff + pParam->BSIM4w0);

          T3 = here->BSIM4eta0 + pParam->BSIM4etab * Vbseff;
          if (T3 < 1.0e-4)
          {   T9 = 1.0 / (3.0 - 2.0e4 * T3);
              T3 = (2.0e-4 - T3) * T9;
              T4 = T9 * T9;
          }
          else
          {   T4 = 1.0;
          }
          dDIBL_Sft_dVd = T3 * pParam->BSIM4theta0vb0;
          DIBL_Sft = dDIBL_Sft_dVd * Vds;

           Lpe_Vb = sqrt(1.0 + pParam->BSIM4lpeb / Leff);

          Vth = model->BSIM4type * here->BSIM4vth0 + (pParam->BSIM4k1ox * sqrtPhis
              - pParam->BSIM4k1 * pParam->BSIM4sqrtPhi) * Lpe_Vb
              - here->BSIM4k2ox * Vbseff - Delt_vth - T2 + (pParam->BSIM4k3
              + pParam->BSIM4k3b * Vbseff) * Vth_NarrowW + T1 - DIBL_Sft;

          dVth_dVb = Lpe_Vb * pParam->BSIM4k1ox * dsqrtPhis_dVb - here->BSIM4k2ox
                   - dDelt_vth_dVb - dT2_dVb + pParam->BSIM4k3b * Vth_NarrowW
                   - pParam->BSIM4etab * Vds * pParam->BSIM4theta0vb0 * T4
                   + pParam->BSIM4kt2 * TempRatio;
          dVth_dVd = -dDIBL_Sft_dVd;


          /* Calculate n */
          tmp1 = epssub / Xdep;
          here->BSIM4nstar = model->BSIM4vtm / Charge_q * (model->BSIM4coxe
                           + tmp1 + pParam->BSIM4cit);  
          tmp2 = pParam->BSIM4nfactor * tmp1;
          tmp3 = pParam->BSIM4cdsc + pParam->BSIM4cdscb * Vbseff
               + pParam->BSIM4cdscd * Vds;
          tmp4 = (tmp2 + tmp3 * Theta0 + pParam->BSIM4cit) / model->BSIM4coxe;
          if (tmp4 >= -0.5)
          {   n = 1.0 + tmp4;
              dn_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb
                     + pParam->BSIM4cdscb * Theta0) / model->BSIM4coxe;
              dn_dVd = pParam->BSIM4cdscd * Theta0 / model->BSIM4coxe;
          }
          else
          {   T0 = 1.0 / (3.0 + 8.0 * tmp4);
              n = (1.0 + 3.0 * tmp4) * T0;
              T0 *= T0;
              dn_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb
                     + pParam->BSIM4cdscb * Theta0) / model->BSIM4coxe * T0;
              dn_dVd = pParam->BSIM4cdscd * Theta0 / model->BSIM4coxe * T0;
          }


          /* Vth correction for Pocket implant */
           if (pParam->BSIM4dvtp0 > 0.0)
          {   T0 = -pParam->BSIM4dvtp1 * Vds;
              if (T0 < -EXP_THRESHOLD)
              {   T2 = MIN_EXP;
                  dT2_dVd = 0.0;
              }
              else
              {   T2 = exp(T0);
                  dT2_dVd = -pParam->BSIM4dvtp1 * T2;
              }

              T3 = Leff + pParam->BSIM4dvtp0 * (1.0 + T2);
              dT3_dVd = pParam->BSIM4dvtp0 * dT2_dVd;
              if (model->BSIM4tempMod < 2)
              {
                T4 = Vtm * log(Leff / T3);
                dT4_dVd = -Vtm * dT3_dVd / T3;
              }
              else
              {
                T4 = model->BSIM4vtm0 * log(Leff / T3);
                dT4_dVd = -model->BSIM4vtm0 * dT3_dVd / T3;
              }
              dDITS_Sft_dVd = dn_dVd * T4 + n * dT4_dVd;
              dDITS_Sft_dVb = T4 * dn_dVb;

              Vth -= n * T4;
              dVth_dVd -= dDITS_Sft_dVd;
              dVth_dVb -= dDITS_Sft_dVb;
          }
        
        /* v4.7 DITS_SFT2  */
        if ((pParam->BSIM4dvtp4  == 0.0) || (pParam->BSIM4dvtp2factor == 0.0)) {
          T0 = 0.0;
            DITS_Sft2 = 0.0;   
        }
        else 
        {
          //T0 = exp(2.0 * pParam->BSIM4dvtp4 * Vds);   /* beta code */
          T1 = 2.0 * pParam->BSIM4dvtp4 * Vds;
          DEXP(T1, T0, T10);
            DITS_Sft2 = pParam->BSIM4dvtp2factor * (T0-1) / (T0+1);   
          //dDITS_Sft2_dVd = pParam->BSIM4dvtp2factor * pParam->BSIM4dvtp4 * 4.0 * T0 / ((T0+1) * (T0+1));   /* beta code */
          dDITS_Sft2_dVd = pParam->BSIM4dvtp2factor * pParam->BSIM4dvtp4 * 4.0 * T10 / ((T0+1) * (T0+1));
          Vth -= DITS_Sft2;
          dVth_dVd -= dDITS_Sft2_dVd;
        }
        
        

          here->BSIM4von = Vth;

          
          /* Poly Gate Si Depletion Effect */
          T0 = here->BSIM4vfb + pParam->BSIM4phi;
          if(model->BSIM4mtrlMod == 0)                 
            T1 = EPSSI;
          else
            T1 = model->BSIM4epsrgate * EPS0;


              BSIM4polyDepletion(T0, pParam->BSIM4ngate, T1, model->BSIM4coxe, vgs, &vgs_eff, &dvgs_eff_dvg);

              BSIM4polyDepletion(T0, pParam->BSIM4ngate, T1, model->BSIM4coxe, vgd, &vgd_eff, &dvgd_eff_dvg);
              
              if(here->BSIM4mode>0) {
                      Vgs_eff = vgs_eff;
                      dVgs_eff_dVg = dvgs_eff_dvg;
              } else {
                      Vgs_eff = vgd_eff;
                      dVgs_eff_dVg = dvgd_eff_dvg;
              }
              here->BSIM4vgs_eff = vgs_eff;
              here->BSIM4vgd_eff = vgd_eff;
              here->BSIM4dvgs_eff_dvg = dvgs_eff_dvg;
              here->BSIM4dvgd_eff_dvg = dvgd_eff_dvg;


          Vgst = Vgs_eff - Vth;

          /* Calculate Vgsteff */
          T0 = n * Vtm;
          T1 = pParam->BSIM4mstar * Vgst;
          T2 = T1 / T0;
          if (T2 > EXP_THRESHOLD)
          {   T10 = T1;
              dT10_dVg = pParam->BSIM4mstar * dVgs_eff_dVg;
              dT10_dVd = -dVth_dVd * pParam->BSIM4mstar;
              dT10_dVb = -dVth_dVb * pParam->BSIM4mstar;
          }
          else if (T2 < -EXP_THRESHOLD)
          {   T10 = Vtm * log(1.0 + MIN_EXP);
              dT10_dVg = 0.0;
              dT10_dVd = T10 * dn_dVd;
              dT10_dVb = T10 * dn_dVb;
              T10 *= n;
          }
          else
          {   ExpVgst = exp(T2);
              T3 = Vtm * log(1.0 + ExpVgst);
              T10 = n * T3;
              dT10_dVg = pParam->BSIM4mstar * ExpVgst / (1.0 + ExpVgst);
              dT10_dVb = T3 * dn_dVb - dT10_dVg * (dVth_dVb + Vgst * dn_dVb / n);
              dT10_dVd = T3 * dn_dVd - dT10_dVg * (dVth_dVd + Vgst * dn_dVd / n);
              dT10_dVg *= dVgs_eff_dVg;
          }

          T1 = pParam->BSIM4voffcbn - (1.0 - pParam->BSIM4mstar) * Vgst;
          T2 = T1 / T0;
          if (T2 < -EXP_THRESHOLD)
          {   T3 = model->BSIM4coxe * MIN_EXP / pParam->BSIM4cdep0;
              T9 = pParam->BSIM4mstar + T3 * n;
              dT9_dVg = 0.0;
              dT9_dVd = dn_dVd * T3;
              dT9_dVb = dn_dVb * T3;
          }
          else if (T2 > EXP_THRESHOLD)
          {   T3 = model->BSIM4coxe * MAX_EXP / pParam->BSIM4cdep0;
              T9 = pParam->BSIM4mstar + T3 * n;
              dT9_dVg = 0.0;
              dT9_dVd = dn_dVd * T3;
              dT9_dVb = dn_dVb * T3;
          }
          else
          {   ExpVgst = exp(T2);
              T3 = model->BSIM4coxe / pParam->BSIM4cdep0;
              T4 = T3 * ExpVgst;
              T5 = T1 * T4 / T0;
              T9 = pParam->BSIM4mstar + n * T4;
              dT9_dVg = T3 * (pParam->BSIM4mstar - 1.0) * ExpVgst / Vtm;
              dT9_dVb = T4 * dn_dVb - dT9_dVg * dVth_dVb - T5 * dn_dVb;
              dT9_dVd = T4 * dn_dVd - dT9_dVg * dVth_dVd - T5 * dn_dVd;
              dT9_dVg *= dVgs_eff_dVg;
          }
          here->BSIM4Vgsteff = Vgsteff = T10 / T9;
          T11 = T9 * T9;
          dVgsteff_dVg = (T9 * dT10_dVg - T10 * dT9_dVg) / T11;
          dVgsteff_dVd = (T9 * dT10_dVd - T10 * dT9_dVd) / T11;
          dVgsteff_dVb = (T9 * dT10_dVb - T10 * dT9_dVb) / T11;

          /* Calculate Effective Channel Geometry */
          T9 = sqrtPhis - pParam->BSIM4sqrtPhi;
          Weff = pParam->BSIM4weff - 2.0 * (pParam->BSIM4dwg * Vgsteff 
