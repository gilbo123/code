import Vue from 'vue'
// import VueFrappe from 'vue2-frappe';
import Chart from 'vue2-frappe'
import App from './App.vue'

Vue.use(Chart)

Vue.config.productionTip = false

new Vue({
  render: h => h(App),
}).$mount('#app')
